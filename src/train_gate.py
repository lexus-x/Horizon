import os
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from replan_gate import ReplanGate, PrefixFeatureExtractor

HARNESS = Path("/home/user/Desktop/vla_projects/RESEARCH/active/_harness-faithful-smolvla")
CKPT = str(HARNESS / "ckpt_smolvla_metaworld")

TASKS = {
    "push-v3": ("Push the puck to a goal", range(2000, 2100)),
    "plate-slide-v3": ("Slide a plate into a cabinet", range(1550, 1600)),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="push-v3", choices=list(TASKS))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=0.05)
    args = ap.parse_args()

    nl, ep_range = TASKS[args.task]
    eps = list(ep_range)
    out_path = args.out or f"../results/gate_{args.task}.pt"
    
    print(f"=== Train Replan-Gate on {args.task} for {args.steps} steps ===")

    policy = SmolVLAPolicy.from_pretrained(CKPT)
    policy.to(args.device)
    policy.eval()

    # Wrap to extract prefix features
    extractor = PrefixFeatureExtractor(policy, pool="mean")
    
    chunk_size = policy.config.chunk_size

    fps = 80
    dt = {
        "action": [i / fps for i in range(chunk_size)],
        "observation.image": [0.0],
        "observation.state": [0.0],
    }
    ds = LeRobotDataset("lerobot/metaworld_mt50", episodes=eps, delta_timestamps=dt)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=CKPT,
        preprocessor_overrides={"device_processor": {"device": args.device},
                                "rename_observations_processor": {"rename_map": {}}},
    )

    gate = None
    opt = None
    bce = nn.BCELoss()

    step = 0
    t0 = time.time()
    
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
                
            batch = pre(batch)
            gt_action = batch["action"].clone().to(args.device) # (B, S, A)
            
            # Predict action chunk using policy (open loop)
            with torch.no_grad():
                from lerobot.policies.utils import populate_queues
                from lerobot.utils.constants import ACTION
                policy._queues = populate_queues(policy._queues, batch, exclude_keys=[ACTION])
                pred_action = policy._get_action_chunk(dict(batch), noise=None) # (B, S, A)
                
            # Compute label: replan if L2 error > threshold
            error = (pred_action - gt_action).norm(dim=-1) # (B, S)
            labels = (error > args.threshold).float() # (B, S)
            
            B, S = labels.shape
            feat = extractor.last # (B, D)
            if feat is None:
                print("Extractor failed to get feature!")
                feat = torch.zeros(B, 960, device=args.device)
            
            if gate is None:
                dim = feat.shape[-1]
                gate = ReplanGate(feature_dim=dim, hidden_dim=128).to(args.device)
                opt = torch.optim.Adam(gate.parameters(), lr=args.lr)
                print(f"Initialized gate with feature_dim {dim}")
            
            loss = 0
            opt.zero_grad()
            for t in range(S):
                step_idx = torch.full((B, 1), t / S, device=args.device)
                prob = gate(feat, step_idx).squeeze(-1) # (B,)
                loss += bce(prob, labels[:, t])
                
            loss = loss / S
            loss.backward()
            opt.step()
            
            if step % 10 == 0:
                print(f"step {step:3d}  loss={loss.item():.4f}")
            step += 1
            
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(gate.state_dict(), out_path)
    print(f"Saved gate to {out_path}")

if __name__ == "__main__":
    main()
