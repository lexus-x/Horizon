#!/usr/bin/env python
"""Learned replan-gate for frozen flow-matching SmolVLA (Horizon, Candidate A).

A lightweight head that, from a CHEAP per-step feature of the frozen policy,
predicts whether the current action chunk has gone stale and should be replanned.
The point is to recover the closed-loop success gain (the open-loop execution gap)
at near-1x compute, by replanning *adaptively* instead of at a fixed frequency.

Feature tap (verified by src/test_extraction.py)
-------------------------------------------------
`VLAFlowMatching.sample_actions` fills the prefix KV cache with a single
`vlm_with_expert.forward(..., fill_kv_cache=True)` (modeling_smolvla.py L818)
*before* the 10-step Euler denoise loop (the ~90%-cost part). The prefix VLM
hidden state (B, prefix_len, 960) is the cheap conditional feature. We mean-pool
it to (B, 960).

NOTE: `sample_actions` calls `vlm_with_expert.forward(...)` DIRECTLY, which bypasses
nn.Module forward hooks. So we capture by wrapping the bound `forward` method
(PrefixFeatureExtractor), and for deploy we recompute just the prefix half with no
denoise loop (compute_prefix_feature) so the gate adds ~near-0 compute.
"""
import torch
import torch.nn as nn

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    ACTION,
)

FEATURE_DIM = 960  # SmolVLM2-500M prefix hidden size (confirmed by probe)


class ReplanGate(nn.Module):
    """Small MLP head: pooled prefix feature + normalized step -> P(replan)."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, step_frac: torch.Tensor) -> torch.Tensor:
        """features: (B, D) pooled. step_frac: (B, 1) in [0,1]. Returns (B,1) prob."""
        if features.dim() > 2:
            features = features.mean(dim=1)
        x = torch.cat([features, step_frac], dim=-1)
        return torch.sigmoid(self.net(x))


def _pool(h: torch.Tensor, pool: str) -> torch.Tensor:
    """(B, L, D) -> (B, D)."""
    if pool == "last":
        return h[:, -1]
    return h.mean(dim=1)  # default: mean over prefix length


class PrefixFeatureExtractor:
    """Capture the prefix pooled feature DURING a full chunk generation.

    Wraps `policy.model.vlm_with_expert.forward` (which `sample_actions` calls
    directly, so hooks don't fire) and stores the pooled prefix hidden state from
    the fill_kv_cache=True call. Use during data collection, where a full chunk is
    generated anyway. Call `.remove()` to restore the original method.
    """

    def __init__(self, policy, pool: str = "mean"):
        self.mwe = policy.model.vlm_with_expert
        self.pool = pool
        self.last = None  # (B, D)
        self._orig = self.mwe.forward

        def wrapped(*args, **kwargs):
            out = self._orig(*args, **kwargs)
            if kwargs.get("fill_kv_cache"):
                oe = out[0]
                h = oe[0] if isinstance(oe, (list, tuple)) else oe  # (B, L, D)
                self.last = _pool(h, self.pool).detach()
            return out

        self.mwe.forward = wrapped

    def remove(self):
        self.mwe.forward = self._orig


@torch.no_grad()
def compute_prefix_feature(policy, batch: dict, pool: str = "mean") -> torch.Tensor:
    """Cheap prefix-only forward (no denoise loop) -> pooled feature (B, D).

    This is exactly the prefix half of VLAFlowMatching.sample_actions (embed_prefix
    + the single fill_kv_cache=True forward), so it is faithful AND cheap: it skips
    the 10 Euler denoise steps that dominate inference cost. Assumes obs queues are
    already populated (same precondition as policy._get_action_chunk).
    """
    m = policy.model
    # mirror _get_action_chunk's obs stacking from the queues
    for k in list(batch.keys()):
        if k in policy._queues and k != ACTION:
            batch[k] = torch.stack(list(policy._queues[k]), dim=1)
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    outputs_embeds, _ = m.vlm_with_expert.forward(
        attention_mask=att_2d,
        position_ids=pos_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=m.config.use_cache,
        fill_kv_cache=True,
    )
    h = outputs_embeds[0]  # (B, L, D)
    return _pool(h, pool)
