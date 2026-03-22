"""Pseudo-reward signals for RA2 supervised learning.

Adapted from mini-AlphaStar's pseudo_reward.py. In an SL context these are
used as **sample importance weights** to up-weight training samples where
the human player makes critical production/economy decisions, and as
**auxiliary losses** that predict build-order conformity and unit composition.

Two reward signals:
1. Build-order reward  – Levenshtein distance between agent's build sequence
   and the human replay's build sequence.  Encourages learning production
   order patterns.
2. Unit-count reward   – Hamming distance between current unit composition
   and target composition from the replay.  Encourages learning army
   composition patterns.  Time-decayed so that early-game economy decisions
   receive higher weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Time-decay schedule (adapted from mini-AlphaStar)
# ---------------------------------------------------------------------------

# RA2 runs at ~60 ticks per second in fast mode, but the game's "tick"
# counter in our training data is the raw simulation tick.  Chronodivide
# uses 15-tick sample intervals at ~60 fps -> ~4 game-seconds per sample.
# We define decay thresholds in *game ticks* rather than minutes.
#   8 min  ≈ 8 * 60 * 60 = 28_800 ticks  (at speed 6)
#  16 min  ≈ 57_600 ticks
#  24 min  ≈ 86_400 ticks
# These are approximate; the important thing is relative ordering.

_TICK_8_MIN = 28_800
_TICK_16_MIN = 57_600
_TICK_24_MIN = 86_400


def time_decay_scale(tick: int) -> float:
    """Return a decay multiplier based on game tick.

    0-8 min  → 1.0
    8-16 min → 0.5
    16-24 min → 0.25
    >24 min  → 0.0
    """
    if tick > _TICK_24_MIN:
        return 0.0
    if tick > _TICK_16_MIN:
        return 0.25
    if tick > _TICK_8_MIN:
        return 0.5
    return 1.0


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def _levenshtein_distance(seq_a: list[int], seq_b: list[int]) -> int:
    """Compute Levenshtein (edit) distance between two integer sequences."""
    len_a, len_b = len(seq_a), len(seq_b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    # Use single-row DP for memory efficiency.
    prev_row = list(range(len_b + 1))
    for i in range(1, len_a + 1):
        curr_row = [i] + [0] * len_b
        for j in range(1, len_b + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr_row[j] = min(
                curr_row[j - 1] + 1,      # insertion
                prev_row[j] + 1,           # deletion
                prev_row[j - 1] + cost,    # substitution
            )
        prev_row = curr_row
    return prev_row[len_b]


def _hamming_distance(vec_a: list[int], vec_b: list[int]) -> int:
    """Compute Hamming distance between two equal-length integer vectors.

    If lengths differ, the shorter one is zero-padded.
    """
    max_len = max(len(vec_a), len(vec_b))
    dist = 0
    for i in range(max_len):
        a = vec_a[i] if i < len(vec_a) else 0
        b = vec_b[i] if i < len(vec_b) else 0
        if a != b:
            dist += 1
    return dist


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def build_order_reward(
    agent_build_order: list[int],
    target_build_order: list[int],
) -> float:
    """Compute build-order pseudo-reward using Levenshtein distance.

    Returns a value in [-0.8, 0] where 0 means perfect match.
    Adapted from mini-AlphaStar: reward = -min(dist², 50) / 50 * 0.8
    """
    dist = _levenshtein_distance(agent_build_order, target_build_order)
    cost = min(dist * dist, 50) / 50.0 * 0.8
    return -cost


def unit_counts_reward(
    agent_counts: list[int],
    target_counts: list[int],
    tick: int = 0,
) -> float:
    """Compute unit-count pseudo-reward using Hamming distance with time decay.

    Returns a negative value (penalty) that decays to 0 after 24 minutes.
    Adapted from mini-AlphaStar.
    """
    dist = _hamming_distance(agent_counts, target_counts)
    scale = time_decay_scale(tick)
    return -dist * scale


# ---------------------------------------------------------------------------
# Batch-level sample importance weights
# ---------------------------------------------------------------------------

# Scale factors to keep pseudo-rewards from dominating the loss.
_SCALE_LEVENSHTEIN = 0.05
_SCALE_HAMMING = 0.05

# Action family IDs for production-related actions.  These get bonus weight.
# (Indices match LABEL_LAYOUT_V2_ACTION_FAMILIES or the v1 action dict.)
_PRODUCTION_FAMILY_NAMES = {"Queue", "PlaceBuilding"}


@dataclass(frozen=True)
class PseudoRewardConfig:
    """Configuration for pseudo-reward sample weighting."""

    # Whether to enable pseudo-reward weighting at all.
    enabled: bool = False

    # Weight multiplier for production-action samples (Queue, PlaceBuilding).
    # Values > 1.0 up-weight these samples relative to others.
    production_action_boost: float = 3.0

    # Weight multiplier for non-Noop action samples.
    non_noop_boost: float = 1.5

    # Levenshtein reward contribution scale.
    build_order_scale: float = _SCALE_LEVENSHTEIN

    # Hamming reward contribution scale.
    unit_count_scale: float = _SCALE_HAMMING

    # Minimum sample weight (prevents zero-weighting).
    min_weight: float = 0.2


def compute_sample_importance_weights(
    batch: dict[str, Any],
    config: PseudoRewardConfig,
    *,
    action_family_names: list[str] | None = None,
    noop_family_index: int = 0,
) -> torch.Tensor:
    """Compute per-sample importance weights for the SL training loss.

    Weights are based on:
    1. Action type: production actions get `production_action_boost`, non-Noop
       gets `non_noop_boost`, Noop gets 1.0.
    2. Build-order conformity: samples from replays with good build orders
       get slightly higher weight (future extension).

    Returns a 1-D tensor of shape [batch_size] (or [batch_size * seq_len]
    for sequence windows) with values >= config.min_weight.
    """
    targets = batch.get("training_targets", {})

    # Determine action family per sample.
    action_family_one_hot = targets.get("actionFamilyOneHot") or targets.get("actionTypeOneHot")
    if action_family_one_hot is None:
        # Can't determine action type – return uniform weights.
        masks = batch.get("training_masks", {})
        first_mask_key = next(iter(masks), None)
        if first_mask_key is not None:
            batch_size = masks[first_mask_key].reshape(-1).shape[0]
        else:
            batch_size = 1
        return torch.ones(batch_size, dtype=torch.float32)

    # Flatten [batch, (seq_len,) num_classes] -> [N, num_classes]
    flat_one_hot = action_family_one_hot.reshape(-1, action_family_one_hot.shape[-1])
    num_samples = flat_one_hot.shape[0]
    family_indices = torch.argmax(flat_one_hot.to(torch.float32), dim=-1)  # [N]

    weights = torch.ones(num_samples, dtype=torch.float32, device=family_indices.device)

    # Apply non-Noop boost.
    is_not_noop = family_indices != noop_family_index
    weights = torch.where(is_not_noop, weights * config.non_noop_boost, weights)

    # Apply production-action boost.
    if action_family_names is not None:
        production_indices = torch.tensor(
            [i for i, name in enumerate(action_family_names) if name in _PRODUCTION_FAMILY_NAMES],
            dtype=torch.long,
            device=family_indices.device,
        )
        if production_indices.numel() > 0:
            is_production = torch.zeros(num_samples, dtype=torch.bool, device=family_indices.device)
            for idx in production_indices:
                is_production |= family_indices == idx
            weights = torch.where(is_production, weights * config.production_action_boost, weights)

    # Clamp to minimum.
    weights = torch.clamp(weights, min=config.min_weight)

    return weights


def apply_importance_weights_to_loss(
    per_sample_loss: torch.Tensor,
    importance_weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply importance weights to a per-sample loss tensor.

    per_sample_loss: [N] unreduced losses per sample.
    importance_weights: [N] importance weights.
    mask: [N] bool mask of valid samples.

    Returns: scalar weighted mean loss.
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=per_sample_loss.device, requires_grad=True)
    weighted = per_sample_loss * importance_weights
    return (weighted * mask.to(weighted.dtype)).sum() / mask.to(weighted.dtype).sum()


# ---------------------------------------------------------------------------
# Auxiliary loss: Build-order prediction head
# ---------------------------------------------------------------------------

class BuildOrderPredictionHead(torch.nn.Module):
    """Auxiliary head that predicts the next item in the build order.

    Given the fused latent state, predicts which object should be built next.
    This provides additional gradient signal for production planning.
    """

    def __init__(self, latent_dim: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(latent_dim, vocab_size),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return logits of shape [..., vocab_size]."""
        return self.projection(latent)


class CompositionPredictionHead(torch.nn.Module):
    """Auxiliary head that predicts target unit composition.

    Given the fused latent state, predicts the count of each unit type
    the player should aim for.  Trained against future composition
    snapshots from the replay.
    """

    def __init__(self, latent_dim: int, composition_size: int, dropout: float = 0.1):
        super().__init__()
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(latent_dim, composition_size),
            torch.nn.ReLU(),  # Counts are non-negative.
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return predicted composition of shape [..., composition_size]."""
        return self.projection(latent)


# ---------------------------------------------------------------------------
# Auxiliary losses
# ---------------------------------------------------------------------------

def build_order_prediction_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss for next build-order item prediction.

    logits: [..., vocab_size] from BuildOrderPredictionHead.
    target_ids: [...] integer IDs of the next build-order item.
    mask: [...] bool mask where target is valid.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = target_ids.reshape(-1).to(torch.long)
    flat_mask = mask.reshape(-1).to(torch.bool)

    if not flat_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    per_sample = loss_fn(flat_logits, flat_targets)
    return (per_sample * flat_mask.to(per_sample.dtype)).sum() / flat_mask.to(per_sample.dtype).sum()


def composition_prediction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 loss for unit composition prediction.

    predicted: [..., composition_size] from CompositionPredictionHead.
    target: [..., composition_size] actual composition counts.
    mask: [...] bool mask where target is valid.
    """
    flat_predicted = predicted.reshape(-1, predicted.shape[-1])
    flat_target = target.reshape(-1, target.shape[-1]).to(flat_predicted.dtype)
    flat_mask = mask.reshape(-1).to(torch.bool)

    if not flat_mask.any():
        return torch.tensor(0.0, device=predicted.device, requires_grad=True)

    per_sample = torch.nn.functional.smooth_l1_loss(
        flat_predicted, flat_target, reduction="none",
    ).mean(dim=-1)  # Average over composition dimensions.
    return (per_sample * flat_mask.to(per_sample.dtype)).sum() / flat_mask.to(per_sample.dtype).sum()
