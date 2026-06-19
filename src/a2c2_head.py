#!/usr/bin/env python
"""A2C2 per-step correction head (Sendai et al., arXiv:2509.23224), adapted for
frozen SmolVLA on MetaWorld.

The base VLA emits a 50-step open-loop chunk once per 50 steps (1x VLM cost). This
tiny MLP then runs EVERY control step and corrects the (stale) base action using
the LATEST proprioceptive state + the base action + the action's index within the
chunk. Output is a per-step correction added to the base action.

Key point: the head does NOT re-run the VLM. It distills "re-observe and act" into
a cheap proprio-conditioned residual -- restoring closed-loop reactivity at ~1x
compute, sidestepping both the dead replan-gate (no when-to-replan signal needed:
it corrects every step) and open-loop trajectory distillation (it keeps live obs).
Everything is in the policy's normalized action space.
"""
import torch
import torch.nn as nn


class A2C2Head(nn.Module):
    def __init__(self, state_dim: int = 4, action_dim: int = 4, hidden: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: torch.Tensor, base_action: torch.Tensor,
                pos_frac: torch.Tensor) -> torch.Tensor:
        """state (B,Ds), base_action (B,Da), pos_frac (B,1) in [0,1].
        Returns correction (B,Da) to ADD to base_action."""
        x = torch.cat([state, base_action, pos_frac], dim=-1)
        return self.net(x)
