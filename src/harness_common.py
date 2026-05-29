#!/usr/bin/env python3
"""Shared harness utilities for VLA research projects (07-ILC, 09-CSD, etc.).

Provides:
- build_env_and_policy():  factored env+policy constructor from cascade_eval.py
- PerStepLogger:           callback-based per-step data collector
- get_ee_position():       EE Cartesian position from MetaWorld privileged state
- get_task_progress():     graded progress g_t ∈ [0,1] from MetaWorld state
- run_rollouts_with_logging(): rollout loop that calls a per-step hook
"""

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from lerobot.envs.configs import MetaworldEnv as MetaworldEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts.lerobot_eval import eval_policy


# ---------------------------------------------------------------------------
# MetaWorld privileged-state extraction
# ---------------------------------------------------------------------------

def get_ee_position_from_obs(obs_state: np.ndarray) -> np.ndarray:
    """Extract EE (TCP) Cartesian position from MetaWorld's full obs vector.

    MetaWorld obs layout (fully-observable, 39-dim):
      [0:3]   = hand_pos (TCP x,y,z)
      [3]     = gripper_open
      [4:7]   = obj_pos
      [36:39] = goal

    Args:
        obs_state: numpy array of shape (..., >=3), the full MetaWorld obs.

    Returns:
        TCP position [x, y, z] as numpy array of shape (3,).
    """
    return np.asarray(obs_state[..., :3], dtype=np.float64)


def get_goal_position_from_obs(obs_state: np.ndarray) -> np.ndarray:
    """Extract goal position from MetaWorld's full obs vector.

    Args:
        obs_state: numpy array of shape (..., 39), the full MetaWorld obs.

    Returns:
        Goal position [x, y, z] as numpy array of shape (3,).
    """
    return np.asarray(obs_state[..., 36:39], dtype=np.float64)


def get_obj_position_from_obs(obs_state: np.ndarray) -> np.ndarray:
    """Extract object position from MetaWorld's full obs vector.

    Args:
        obs_state: numpy array of shape (..., >=7), the full MetaWorld obs.

    Returns:
        Object position [x, y, z] as numpy array of shape (3,).
    """
    return np.asarray(obs_state[..., 4:7], dtype=np.float64)


def compute_graded_progress(obs_state: np.ndarray, task_type: str = "reach") -> float:
    """Compute a graded progress metric g_t ∈ [0, 1] from MetaWorld state.

    For reach tasks: progress = 1 - (dist_ee_to_goal / max_dist), clipped.
    For manipulation tasks: weighted combination of gripper→object and
    object→goal distances.

    Args:
        obs_state: full MetaWorld observation (39-dim).
        task_type: "reach" or "manipulate".

    Returns:
        Float in [0, 1] where 1 = task completed.
    """
    ee = get_ee_position_from_obs(obs_state)
    goal = get_goal_position_from_obs(obs_state)

    if task_type == "reach":
        dist = np.linalg.norm(ee - goal)
        # MetaWorld workspace is roughly 0.4m across; normalize
        max_dist = 0.4
        progress = 1.0 - min(dist / max_dist, 1.0)
        return float(progress)

    # Manipulation: two-phase progress
    obj = get_obj_position_from_obs(obs_state)
    dist_ee_obj = np.linalg.norm(ee - obj)
    dist_obj_goal = np.linalg.norm(obj - goal)
    max_dist = 0.4
    # Phase 1: approach object (weight 0.3), Phase 2: move object to goal (weight 0.7)
    approach = 1.0 - min(dist_ee_obj / max_dist, 1.0)
    place = 1.0 - min(dist_obj_goal / max_dist, 1.0)
    return float(0.3 * approach + 0.7 * place)


# ---------------------------------------------------------------------------
# Environment + policy builder (factored from cascade_eval.py)
# ---------------------------------------------------------------------------

def build_env_and_policy(
    ckpt: str,
    task: str = "reach-v3",
    batch_size: int = 1,
    device: str = "cuda",
    policy_cls=None,
):
    """Build MetaWorld vectorized env and SmolVLA policy with correct processors.

    Args:
        ckpt: path to the local smolvla_metaworld checkpoint.
        task: MetaWorld task name (e.g., "reach-v3").
        batch_size: number of parallel envs.
        device: torch device.
        policy_cls: optional SmolVLAPolicy subclass to use instead of base.

    Returns:
        (vec_env, policy, preprocessor, postprocessor, env_pre, env_post)
    """
    env_cfg = MetaworldEnvConfig(task=task, obs_type="pixels_agent_pos")
    envs = make_env(env_cfg, n_envs=batch_size, use_async_envs=False)
    vec = envs[task][0]

    if policy_cls is None:
        policy_cls = SmolVLAPolicy

    policy = policy_cls.from_pretrained(ckpt)
    policy.to(device)
    policy.eval()

    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=ckpt,
        preprocessor_overrides={
            "device_processor": {"device": device},
            # Disable stale rename map (documented faithfulness fix)
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy.config
    )
    return vec, policy, pre, post, env_pre, env_post


# ---------------------------------------------------------------------------
# Per-step data logging
# ---------------------------------------------------------------------------

class PerStepLogger:
    """Collects per-step data during rollouts for offline analysis.

    Usage:
        logger = PerStepLogger()
        logger.new_episode(seed=42)
        for step in rollout:
            logger.log_step(
                ee_pos=..., goal_pos=..., action=...,
                reward=..., success=..., progress=...,
                extras={"chunk_variance": ...}
            )
        logger.end_episode(success=True)
        logger.save("results.json")
    """

    def __init__(self):
        self.episodes = []
        self._current = None

    def new_episode(self, seed: int = -1, episode_ix: int = -1, **meta):
        self._current = {
            "seed": seed,
            "episode_ix": episode_ix,
            "meta": meta,
            "steps": [],
            "success": False,
        }

    def log_step(
        self,
        ee_pos=None,
        goal_pos=None,
        action=None,
        reward=None,
        success=None,
        progress=None,
        extras=None,
    ):
        step = {}
        if ee_pos is not None:
            step["ee_pos"] = np.asarray(ee_pos, dtype=np.float64).tolist()
        if goal_pos is not None:
            step["goal_pos"] = np.asarray(goal_pos, dtype=np.float64).tolist()
        if action is not None:
            step["action"] = np.asarray(action, dtype=np.float64).tolist()
        if reward is not None:
            step["reward"] = float(reward)
        if success is not None:
            step["success"] = bool(success)
        if progress is not None:
            step["progress"] = float(progress)
        if extras is not None:
            # Ensure all values are JSON-serializable
            step["extras"] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else
                    v.item() if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in extras.items()
            }
        self._current["steps"].append(step)

    def end_episode(self, success: bool = False):
        self._current["success"] = success
        self.episodes.append(self._current)
        self._current = None

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(
                {
                    "n_episodes": len(self.episodes),
                    "episodes": self.episodes,
                },
                f,
                indent=2,
            )
        print(f"Saved {len(self.episodes)} episodes to {p}")

    @staticmethod
    def load(path: str) -> dict:
        with open(path) as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def save_results(path: str, data: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test():
    """Quick smoke test: build env+policy, run 2 episodes, check EE extraction."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(
        Path(__file__).parent / "ckpt_smolvla_metaworld"))
    ap.add_argument("--task", default="reach-v3")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"Building env+policy for {args.task}...")
    vec, policy, pre, post, env_pre, env_post = build_env_and_policy(
        args.ckpt, args.task, batch_size=1, device=args.device
    )
    print("  ✓ Build succeeded")

    # Quick rollout via eval_policy
    info = eval_policy(
        env=vec,
        policy=policy,
        env_preprocessor=env_pre,
        env_postprocessor=env_post,
        preprocessor=pre,
        postprocessor=post,
        n_episodes=2,
        max_episodes_rendered=0,
        videos_dir=None,
        start_seed=42,
    )
    agg = info["aggregated"]
    print(f"  ✓ Rollout: success={agg['pc_success']}%, "
          f"avg_max_reward={agg['avg_max_reward']:.2f}")

    # Test privileged state extraction
    obs = np.random.randn(39)
    ee = get_ee_position_from_obs(obs)
    goal = get_goal_position_from_obs(obs)
    progress = compute_graded_progress(obs, "reach")
    print(f"  ✓ State extraction: ee={ee[:3]}, goal={goal[:3]}, "
          f"progress={progress:.3f}")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    _smoke_test()
