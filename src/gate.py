#!/usr/bin/env python
"""Horizon — the trained adaptive replan-gate on a FROZEN flow-matching VLA.

This is the proposed contribution itself (previously only a scaffold). The frozen
SmolVLA executes 50-step action chunks open-loop; replanning more often (closed
loop) lifts success but costs ~5x inference. A *learned when-to-replan gate* aims
to recover that gain at a far smaller replan budget — the falsifiable bar (the
"kill-gate") is that the learned gate must BEAT fixed-frequency replanning at a
MATCHED replan budget (smarter *when*, not just *more often*). If it only ties
fixed-frequency, it collapses to the known receding-horizon knob.

Pipeline (one file, three subcommands)
--------------------------------------
  collect : roll out the frozen policy under a randomized replan schedule; at each
            step record cheap features x_t (proprio drift / open-loop tracking
            residual / phase / committed action) and a regression target
            d_t = || E[fresh replan action now] - committed_stale_action ||.
            The target is the *staleness* of the currently-executing chunk: how
            far the action we are about to commit has drifted from what the policy
            would freshly choose. The reference E[.] averages M fresh draws to
            damp flow sampling noise.
  train   : fit a tiny MLP  x_t -> d_t  (standardized features). Reports held-out
            predictive correlation (is there any signal to gate on at all?) and
            stores prediction quantiles so eval can pick thresholds by budget.
  eval    : run an arm through lerobot's faithful eval_policy and log the per-arm
            average replan budget + per-episode success:
              base   : open-loop h=50 (stock).
              fixed  : fixed-frequency replanning every H steps (the known knob).
              gate   : replan only when MLP(x_t) > tau (cap at chunk_size).
            Deploy-time the gate runs on cheap proprio features only (no extra VLM
            forward), so a non-firing step costs ~0 — near-1x compute.

All arms reuse harness_common.build_env_and_policy (the verified-faithful env +
official pre/post processors), so success numbers stay bit-for-bit comparable to
the base survey. Replan counting is exact because eval runs at batch_size=1 (one
episode per rollout; policy.reset() per episode — same trick as dispersion_probe).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import populate_queues
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.constants import ACTION, OBS_STATE

from harness_common import build_env_and_policy

STATE_DIM = 4    # MetaWorld agent_pos (hand x,y,z + gripper), normalized
ACT_DIM = 4
FEAT_DIM = 1 + 5 * STATE_DIM  # phase + a_committed + s_t + drift + track_resid + vel = 21


# --------------------------------------------------------------------------- #
# Feature extraction (shared by collect + gate eval). Cheap: proprio only.
# --------------------------------------------------------------------------- #
def compute_features(committed, k, s_t, s_replan, s_prev, chunk_size):
    """Return (B, FEAT_DIM) gate input from cheap, already-available signals.

    committed : (B, S, A) currently-executing chunk (normalized actions).
    k         : int offset within the committed chunk about to be executed.
    s_t       : (B, STATE_DIM) current normalized proprio state.
    s_replan  : (B, STATE_DIM) proprio state at the last replan.
    s_prev    : (B, STATE_DIM) proprio state one step ago.
    """
    B = s_t.shape[0]
    dev = s_t.device
    kk = min(k, committed.shape[1] - 1)
    phase = torch.full((B, 1), k / chunk_size, device=dev, dtype=s_t.dtype)
    a_k = committed[:, kk, :STATE_DIM]                       # committed action (B,A)
    drift = s_t - s_replan                                   # proprio drift since replan
    cum = (committed[:, :k, :STATE_DIM].sum(dim=1) if k > 0
           else torch.zeros_like(a_k))                       # intended displacement
    track = drift - cum                                      # open-loop tracking residual
    vel = s_t - s_prev                                       # recent proprio velocity
    return torch.cat([phase, a_k, s_t, drift, track, vel], dim=-1)


# --------------------------------------------------------------------------- #
# Gate head
# --------------------------------------------------------------------------- #
class GateMLP(nn.Module):
    def __init__(self, in_dim=FEAT_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# Policy subclass: collect / base / fixed / gate
# --------------------------------------------------------------------------- #
class GatedSmolVLA(SmolVLAPolicy):
    """Frozen SmolVLA with a learned (or fixed) replan schedule."""

    # configured after construction
    mode: str = "base"               # base | fixed | gate | collect
    horizon: int = 50                # exec cap before forced replan
    collect_M: int = 4               # fresh draws averaged for the staleness target
    collect_p: float = 0.2           # randomized adoption prob during collection
    gate_tau: float = 0.0

    def _init_gate_state(self):
        self._committed = None
        self._k = 0
        self._s_replan = None
        self._s_prev = None
        self._ep_replans = 0
        self._started = False
        self._n_replans = 0
        self._replan_log = []
        # gate net (set externally for mode == gate)
        self._gate = getattr(self, "_gate", None)
        self._fmean = getattr(self, "_fmean", None)
        self._fstd = getattr(self, "_fstd", None)
        # collect buffers
        self._feat_buf = getattr(self, "_feat_buf", [])
        self._targ_buf = getattr(self, "_targ_buf", [])
        self._rng = getattr(self, "_rng", np.random.default_rng(0))

    def reset(self):
        super().reset()
        if getattr(self, "_started", False) and self.mode in ("base", "fixed", "gate"):
            self._replan_log.append(self._ep_replans)
        self._committed = None
        self._k = 0
        self._s_replan = None
        self._s_prev = None
        self._ep_replans = 0
        self._started = True

    def flush(self):
        if getattr(self, "_started", False) and self.mode in ("base", "fixed", "gate"):
            self._replan_log.append(self._ep_replans)
            self._started = False

    def _state(self, batch):
        s = batch[OBS_STATE]
        if s.ndim > 2:
            s = s[:, -1, :]
        return s[:, :STATE_DIM].float()

    @torch.no_grad()
    def _gate_score(self, x):
        z = (x - self._fmean) / self._fstd
        return self._gate(z)  # (B,)

    @torch.no_grad()
    def _replan(self, batch, s_t):
        self._committed = self._get_action_chunk(dict(batch), noise=None)  # (B,S,A)
        self._k = 0
        self._s_replan = s_t.clone()
        self._ep_replans += 1
        self._n_replans += 1

    @torch.no_grad()
    def select_action(self, batch, noise=None, **kwargs):
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        s_t = self._state(batch)
        if self._s_prev is None:
            self._s_prev = s_t.clone()

        if self.mode == "collect":
            return self._collect_step(batch, s_t)

        cap = min(self.horizon, self.config.chunk_size)
        need = self._committed is None or self._k >= cap
        if not need and self.mode == "gate":
            x = compute_features(self._committed, self._k, s_t, self._s_replan,
                                 self._s_prev, self.config.chunk_size)
            need = bool((self._gate_score(x) > self.gate_tau).any().item())
        if need:
            self._replan(batch, s_t)

        a = self._committed[:, self._k]
        self._k += 1
        self._s_prev = s_t.clone()
        return a

    @torch.no_grad()
    def _collect_step(self, batch, s_t):
        # M fresh draws -> denoised reference action for "what I'd do if I replanned now"
        draws = torch.stack([self._get_action_chunk(dict(batch), noise=None)
                             for _ in range(self.collect_M)], dim=0)  # (M,B,S,A)
        ref0 = draws[:, :, 0, :].mean(dim=0)  # (B,A)
        fresh_chunk = draws[-1]               # (B,S,A) a valid sample to adopt

        if self._committed is None or self._k >= self.config.chunk_size:
            self._committed = fresh_chunk
            self._k = 0
            self._s_replan = s_t.clone()
            x = compute_features(self._committed, 0, s_t, s_t, self._s_prev,
                                 self.config.chunk_size)
            target = torch.zeros(s_t.shape[0], device=s_t.device)
        else:
            x = compute_features(self._committed, self._k, s_t, self._s_replan,
                                 self._s_prev, self.config.chunk_size)
            target = (ref0 - self._committed[:, self._k, :STATE_DIM]).norm(dim=-1)
            if self._rng.random() < self.collect_p:
                self._committed = fresh_chunk
                self._k = 0
                self._s_replan = s_t.clone()

        self._feat_buf.append(x.detach().cpu().numpy())
        self._targ_buf.append(target.detach().cpu().numpy())

        a = self._committed[:, self._k]
        self._k += 1
        self._s_prev = s_t.clone()
        return a


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def _build(ckpt, task, batch_size, device):
    vec, policy, pre, post, env_pre, env_post = build_env_and_policy(
        ckpt, task=task, batch_size=batch_size, device=device, policy_cls=GatedSmolVLA)
    policy._init_gate_state()
    return vec, policy, pre, post, env_pre, env_post


def cmd_collect(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    vec, policy, pre, post, env_pre, env_post = _build(
        args.ckpt, args.task, batch_size=1, device=args.device)
    policy.mode = "collect"
    policy.collect_M = args.M
    policy.collect_p = args.collect_p
    policy._rng = np.random.default_rng(args.seed)

    eval_policy(env=vec, policy=policy, env_preprocessor=env_pre,
                env_postprocessor=env_post, preprocessor=pre, postprocessor=post,
                n_episodes=args.n_episodes, max_episodes_rendered=0,
                videos_dir=None, start_seed=args.seed)

    feats = np.concatenate(policy._feat_buf, axis=0).astype(np.float32)
    targs = np.concatenate(policy._targ_buf, axis=0).astype(np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, feats=feats, targs=targs, task=args.task)
    print(f"COLLECT task={args.task} n={feats.shape[0]} feat_dim={feats.shape[1]} "
          f"target[mean={targs.mean():.4f} p50={np.median(targs):.4f} "
          f"p90={np.quantile(targs,0.9):.4f}] -> {out}", flush=True)


def cmd_train(args):
    torch.manual_seed(args.seed)
    d = np.load(args.data)
    X = torch.from_numpy(d["feats"]).float()
    y = torch.from_numpy(d["targs"]).float()
    n = X.shape[0]
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g)
    X, y = X[perm], y[perm]
    n_val = max(1, int(0.15 * n))
    Xtr, ytr, Xva, yva = X[n_val:], y[n_val:], X[:n_val], y[:n_val]

    fmean = Xtr.mean(0, keepdim=True)
    fstd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Ztr, Zva = (Xtr - fmean) / fstd, (Xva - fmean) / fstd

    dev = args.device
    net = GateMLP(in_dim=X.shape[1], hidden=args.hidden).to(dev)
    Ztr, ytr_d = Ztr.to(dev), ytr.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    bs = min(args.batch_size, Ztr.shape[0])
    for step in range(args.steps):
        idx = torch.randint(0, Ztr.shape[0], (bs,), device=dev)
        pred = net(Ztr[idx])
        loss = ((pred - ytr_d[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.log_every == 0:
            print(f"step {step:5d} mse={loss.item():.5f}", flush=True)

    net.eval()
    with torch.no_grad():
        pva = net(Zva.to(dev)).cpu()
        ptr = net(Ztr).cpu()
    # held-out predictive correlation: is there any staleness signal to gate on?
    if yva.std() > 1e-8 and pva.std() > 1e-8:
        corr = float(np.corrcoef(pva.numpy(), yva.numpy())[0, 1])
    else:
        corr = 0.0
    quants = {str(q): float(np.quantile(ptr.numpy(), q))
              for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "in_dim": X.shape[1],
                "hidden": args.hidden, "fmean": fmean, "fstd": fstd,
                "val_corr": corr, "pred_quantiles": quants, "task": str(d["task"])},
               out)
    print(f"TRAIN task={d['task']} val_corr={corr:.3f} n={n} -> {out}", flush=True)
    print("PRED_QUANTILES " + json.dumps(quants), flush=True)


def _load_gate(policy, gate_path, device):
    g = torch.load(gate_path, map_location=device, weights_only=False)
    net = GateMLP(in_dim=g["in_dim"], hidden=g["hidden"]).to(device)
    net.load_state_dict(g["state_dict"]); net.eval()
    policy._gate = net
    policy._fmean = g["fmean"].to(device)
    policy._fstd = g["fstd"].to(device)
    return g


def run_arm(vec, policy, pre, post, env_pre, env_post, *, mode, horizon, tau,
            n_episodes, seed, tau_q=None):
    """Run one eval arm in-process; return its result + per-episode success.

    Arms share start_seed -> episode i is the same initial condition across arms
    (paired for McNemar). torch is re-seeded per arm so flow noise is reproducible.
    """
    policy._init_gate_state()      # resets committed/counters; preserves loaded gate
    policy.mode = mode
    policy.horizon = horizon
    if mode == "gate":
        policy.gate_tau = tau
    torch.manual_seed(seed)
    info = eval_policy(env=vec, policy=policy, env_preprocessor=env_pre,
                       env_postprocessor=env_post, preprocessor=pre, postprocessor=post,
                       n_episodes=n_episodes, max_episodes_rendered=0,
                       videos_dir=None, start_seed=seed)
    policy.flush()
    succ = [bool(e["success"]) for e in info["per_episode"]]
    return {
        "mode": mode, "horizon": horizon,
        "tau": (float(tau) if mode == "gate" else None), "tau_q": tau_q,
        "n_episodes": n_episodes, "seed": seed,
        "pc_success": info["aggregated"]["pc_success"],
        "avg_max_reward": info["aggregated"]["avg_max_reward"],
        "avg_replans": float(np.mean(policy._replan_log)) if policy._replan_log else None,
        "per_episode_success": succ,
        "replan_log": list(policy._replan_log),
    }


def cmd_eval(args):
    np.random.seed(args.seed)
    vec, policy, pre, post, env_pre, env_post = _build(
        args.ckpt, args.task, batch_size=1, device=args.device)
    gmeta = None
    tau = None
    if args.mode == "gate":
        gmeta = _load_gate(policy, args.gate, args.device)
        tau = (args.tau if args.tau is not None
               else gmeta["pred_quantiles"][str(args.tau_q)] if args.tau_q is not None
               else gmeta["pred_quantiles"]["0.5"])
    res = run_arm(vec, policy, pre, post, env_pre, env_post, mode=args.mode,
                  horizon=args.horizon, tau=tau, tau_q=args.tau_q,
                  n_episodes=args.n_episodes, seed=args.seed)
    res["task"] = args.task
    res["val_corr"] = (gmeta["val_corr"] if gmeta else None)
    print("RESULT " + json.dumps({k: v for k, v in res.items()
                                  if k not in ("per_episode_success", "replan_log")}), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"saved -> {args.out}", flush=True)


def cmd_sweep(args):
    """Run a full set of arms for one task in a single process (load model once).

    Arms: base (h=50) + fixed-frequency at each --fixed horizon + gate at each
    --gate_q prediction quantile. All paired on the same seed pool.
    """
    np.random.seed(args.seed)
    vec, policy, pre, post, env_pre, env_post = _build(
        args.ckpt, args.task, batch_size=1, device=args.device)
    gmeta = _load_gate(policy, args.gate, args.device)
    fixed = [int(h) for h in args.fixed.split(",") if h]
    gate_qs = [float(q) for q in args.gate_q.split(",") if q]

    arms = []
    common = dict(n_episodes=args.n_episodes, seed=args.seed)
    print(f"=== SWEEP {args.task} (val_corr={gmeta['val_corr']:.3f}) ===", flush=True)
    arms.append(run_arm(vec, policy, pre, post, env_pre, env_post,
                        mode="base", horizon=50, tau=None, **common))
    print("ARM " + json.dumps({k: arms[-1][k] for k in ("mode", "horizon", "pc_success", "avg_replans")}), flush=True)
    for h in fixed:
        arms.append(run_arm(vec, policy, pre, post, env_pre, env_post,
                            mode="fixed", horizon=h, tau=None, **common))
        print("ARM " + json.dumps({k: arms[-1][k] for k in ("mode", "horizon", "pc_success", "avg_replans")}), flush=True)
    for q in gate_qs:
        tau = gmeta["pred_quantiles"][str(q)]
        arms.append(run_arm(vec, policy, pre, post, env_pre, env_post,
                            mode="gate", horizon=50, tau=tau, tau_q=q, **common))
        print("ARM " + json.dumps({k: arms[-1][k] for k in ("mode", "tau_q", "pc_success", "avg_replans")}), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"task": args.task, "val_corr": gmeta["val_corr"],
                   "n_episodes": args.n_episodes, "seed": args.seed, "arms": arms}, f, indent=2)
    print(f"SWEEP_DONE {args.task} -> {out}", flush=True)


# --------------------------------------------------------------------------- #
# Matched-budget verdict (deterministic stats — the kill-gate)
# --------------------------------------------------------------------------- #
def _mcnemar_exact(a, b):
    """Two-sided exact McNemar on paired binary lists a (arm1), b (arm2).

    Returns (b01, b10, p) where b01 = a wins (a=1,b=0), b10 = b wins.
    p tests H0: symmetric (no difference in discordant pairs).
    """
    from math import comb
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n))
    return b01, b10, p


def _interp(x, xs, ys):
    """Linear interp of y at x. Returns None when x is OUTSIDE the measured
    budget range. Clamping there would falsely credit the gate for spending more
    replans than any fixed arm measured — there is no matched-budget comparison
    to make beyond the densest fixed arm, so such points are not comparable."""
    pts = sorted(zip(xs, ys))
    xs2, ys2 = [p[0] for p in pts], [p[1] for p in pts]
    if x < xs2[0] - 1e-9 or x > xs2[-1] + 1e-9:
        return None
    if x <= xs2[0]:
        return ys2[0]
    if x >= xs2[-1]:
        return ys2[-1]
    for i in range(1, len(xs2)):
        if x <= xs2[i]:
            t = (x - xs2[i - 1]) / (xs2[i] - xs2[i - 1] + 1e-12)
            return ys2[i - 1] + t * (ys2[i] - ys2[i - 1])
    return ys2[-1]


def cmd_verdict(args):
    tasks = {}
    for path in args.sweeps:
        d = json.load(open(path))
        tasks[d["task"]] = d

    report = {"tasks": {}, "summary": None}
    wins = []
    for task, d in tasks.items():
        arms = d["arms"]
        base = next(a for a in arms if a["mode"] == "base")
        fixed = sorted([a for a in arms if a["mode"] == "fixed"], key=lambda a: a["avg_replans"])
        gate = sorted([a for a in arms if a["mode"] == "gate"], key=lambda a: a["avg_replans"])
        fb = [base] + fixed  # fixed-frequency curve incl. base (h=50)
        fx_budget = [a["avg_replans"] for a in fb]
        fx_succ = [a["pc_success"] for a in fb]

        # For each gate arm: compare to fixed-frequency at the SAME replan budget.
        gate_rows = []
        for g in gate:
            bg = g["avg_replans"]
            fixed_interp = _interp(bg, fx_budget, fx_succ)
            in_range = fixed_interp is not None
            delta = round(g["pc_success"] - fixed_interp, 2) if in_range else None
            # nearest actual fixed/base arm for a paired McNemar
            near = min(fb, key=lambda a: abs(a["avg_replans"] - bg))
            b01, b10, p = _mcnemar_exact(g["per_episode_success"], near["per_episode_success"])
            gate_rows.append({
                "tau_q": g["tau_q"], "gate_budget": bg, "gate_success": g["pc_success"],
                "in_budget_range": in_range,
                "fixed_interp_success_at_budget": (round(fixed_interp, 2) if in_range else None),
                "delta_vs_fixed_curve": delta,
                "nearest_fixed": {"mode": near["mode"], "horizon": near["horizon"],
                                  "budget": near["avg_replans"], "success": near["pc_success"]},
                "mcnemar_gate_wins": b01, "mcnemar_fixed_wins": b10, "mcnemar_p": round(p, 4),
            })
        # Only IN-RANGE gate points admit a matched-budget comparison. The kill-gate
        # PASSES only if the best such point beats the fixed curve AND that win is
        # significant under the paired exact McNemar (p<0.05, gate>fixed discordant).
        in_rows = [r for r in gate_rows if r["delta_vs_fixed_curve"] is not None]
        best = max(in_rows, key=lambda r: r["delta_vs_fixed_curve"]) if in_rows else None
        beat = bool(best and best["delta_vs_fixed_curve"] > 0
                    and best["mcnemar_p"] < 0.05
                    and best["mcnemar_gate_wins"] > best["mcnemar_fixed_wins"])
        wins.append(beat)
        report["tasks"][task] = {
            "val_corr": d["val_corr"],
            "fixed_curve": [{"mode": a["mode"], "horizon": a["horizon"],
                             "budget": round(a["avg_replans"], 2), "success": a["pc_success"]} for a in fb],
            "gate_curve": gate_rows,
            "best_gate": best,
            "beats_fixed_at_matched_budget": beat,
        }

    n_beat = sum(wins)
    report["summary"] = (
        f"Gate beats fixed-frequency at matched budget on {n_beat}/{len(wins)} tasks. "
        + ("KILL-GATE PASSED (smarter when-to-replan adds success per replan)."
           if n_beat == len(wins) and len(wins) > 0
           else "KILL-GATE NOT PASSED on all tasks — where it only ties, the gate "
                "collapses to the known receding-horizon knob.")
    )
    print("VERDICT " + json.dumps(report, indent=2), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"saved -> {args.out}", flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    DEF_CKPT = "/home/user/Desktop/vla_projects/RESEARCH/active/_harness-faithful-smolvla/ckpt_smolvla_metaworld"

    c = sub.add_parser("collect")
    c.add_argument("--ckpt", default=DEF_CKPT); c.add_argument("--task", default="push-v3")
    c.add_argument("--n_episodes", type=int, default=24); c.add_argument("--M", type=int, default=4)
    c.add_argument("--collect_p", type=float, default=0.2); c.add_argument("--seed", type=int, default=1000)
    c.add_argument("--device", default="cuda"); c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_collect)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True); t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=4000); t.add_argument("--hidden", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--batch_size", type=int, default=512)
    t.add_argument("--log_every", type=int, default=500); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda"); t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval")
    e.add_argument("--ckpt", default=DEF_CKPT); e.add_argument("--task", default="push-v3")
    e.add_argument("--mode", choices=["base", "fixed", "gate"], default="base")
    e.add_argument("--horizon", type=int, default=50)
    e.add_argument("--gate", default=None); e.add_argument("--tau", type=float, default=None)
    e.add_argument("--tau_q", type=float, default=None)
    e.add_argument("--n_episodes", type=int, default=50); e.add_argument("--seed", type=int, default=1000)
    e.add_argument("--device", default="cuda"); e.add_argument("--out", default=None)
    e.set_defaults(func=cmd_eval)

    s = sub.add_parser("sweep")
    s.add_argument("--ckpt", default=DEF_CKPT); s.add_argument("--task", default="push-v3")
    s.add_argument("--gate", required=True)
    s.add_argument("--fixed", default="30,20,15,10")
    s.add_argument("--gate_q", default="0.3,0.4,0.5,0.6,0.7,0.8")
    s.add_argument("--n_episodes", type=int, default=50); s.add_argument("--seed", type=int, default=1000)
    s.add_argument("--device", default="cuda"); s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_sweep)

    v = sub.add_parser("verdict")
    v.add_argument("--sweeps", nargs="+", required=True)
    v.add_argument("--out", default=None)
    v.set_defaults(func=cmd_verdict)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
