"""Utility helpers for working with Text2Sign checkpoints."""
from __future__ import annotations

from typing import Iterable, Optional


_ARCH_UNKNOWN = "unknown"


def _contains(keys: Iterable[str], substring: str) -> bool:
    """Return True if any checkpoint key contains the given substring."""
    return any(substring in key for key in keys)


def detect_checkpoint_architecture(state_dict_keys: Iterable[str]) -> Optional[str]:
    """Infer the model architecture that produced a checkpoint state dict.

    Args:
        state_dict_keys: Iterable of keys from the checkpoint's ``model_state_dict``.

    Returns:
        The detected architecture string ("unet3d", "vit3d", "dit3d", "vivit", "tinyfusion"),
        or ``"unknown"`` when detection fails.
    """
    keys = list(state_dict_keys)
    if not keys:
        return _ARCH_UNKNOWN

    # ViT3D checkpoints prefix everything with ``model.backbone.vit``.
    if any("model.backbone.vit" in key for key in keys):
        return "vit3d"

    # ViViT checkpoints include dedicated temporal transformer layers.
    if _contains(keys, "model.temporal_layers."):
        return "vivit"

    # TinyFusion wraps a DiT backbone and adds temporal post-processing + AdaLN modulation.
    if _contains(keys, "model.temporal_post") or _contains(keys, "adaLN_modulation"):
        return "tinyfusion"

    # DiT3D checkpoints expose dit_blocks as part of the backbone.
    if _contains(keys, "dit_blocks") or (
        _contains(keys, "model.backbone.x_embedder") and _contains(keys, "model.backbone.y_embedder")
    ):
        return "dit3d"

    # UNet3D checkpoints keep the original init conv blocks / resblocks naming.
    if _contains(keys, "init_conv") or _contains(keys, "encoder_resblocks"):
        return "unet3d"

    return _ARCH_UNKNOWN


__all__ = ["detect_checkpoint_architecture"]
