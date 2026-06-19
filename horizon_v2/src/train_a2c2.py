#!/usr/bin/env python
"""Train the A2C2 correction head: (state, base_action, pos) -> correction, where
target correction = (re-observed action) - (stale open-loop base action). Pure MLP
on normalized actions; no VLM, no base weights touched."""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from a2c2_head import A2C2Head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    d = torch.load(args.data, map_location="cpu")
    state, base_a, pos, target = d["state"], d["base_a"], d["pos_frac"], d["target"]
    corr = target - base_a                      # supervision
    print(f"loaded {state.shape[0]} samples; state{tuple(state.shape)} act{tuple(base_a.shape)} "
          f"| mean|corr|={corr.abs().mean():.4f}", flush=True)

    ds = TensorDataset(state, base_a, pos, corr)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    head = A2C2Head(state_dim=state.shape[-1], action_dim=base_a.shape[-1], hidden=args.hidden).to(args.device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    # baseline: predicting zero correction (i.e. pure open-loop) MSE
    print(f"zero-correction MSE (open-loop baseline) = {(corr**2).mean():.5f}", flush=True)

    step = 0
    while step < args.steps:
        for s, b, p, c in dl:
            if step >= args.steps:
                break
            s, b, p, c = s.to(args.device), b.to(args.device), p.to(args.device), c.to(args.device)
            pred = head(s, b, p)
            loss = mse(pred, c)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 500 == 0:
                print(f"step {step:5d}  loss={loss.item():.5f}", flush=True)
            step += 1
    torch.save({"state_dict": head.state_dict(), "state_dim": state.shape[-1],
                "action_dim": base_a.shape[-1], "hidden": args.hidden}, args.out)
    print(f"saved head -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
