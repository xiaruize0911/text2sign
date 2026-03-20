"""Conditioning ablation presets for Text2Sign experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Tuple

from config import ModelConfig, TrainingConfig


ABLATION_PRESETS: Dict[str, Dict[str, object]] = {
    "frozen_clip": {
        "description": "Frozen CLIP baseline with standard text conditioning.",
        "model": {
            "use_clip_text_encoder": True,
            "text_conditioning_mode": "normal",
            "clip_trainable_layers": 0,
        },
        "training": {},
    },
    "no_text": {
        "description": "No-text control: zero out text embeddings during train/inference.",
        "model": {
            "use_clip_text_encoder": True,
            "text_conditioning_mode": "none",
            "clip_trainable_layers": 0,
        },
        "training": {},
    },
    "random_text": {
        "description": "Random-text control: shuffle text embeddings across the batch.",
        "model": {
            "use_clip_text_encoder": True,
            "text_conditioning_mode": "random",
            "clip_trainable_layers": 0,
        },
        "training": {},
    },
    "clip_finetuned_last2": {
        "description": "Fine-tune the last 2 CLIP text layers for a fairer baseline.",
        "model": {
            "use_clip_text_encoder": True,
            "text_conditioning_mode": "normal",
            "clip_trainable_layers": 2,
        },
        "training": {},
    },
}


def list_ablation_presets() -> Dict[str, str]:
    """Return preset names and descriptions for CLI/help output."""
    return {name: spec["description"] for name, spec in ABLATION_PRESETS.items()}


def apply_ablation_preset(
    model_config: ModelConfig,
    train_config: TrainingConfig,
    preset_name: str,
) -> Tuple[ModelConfig, TrainingConfig, str]:
    """Apply a named conditioning ablation preset to config dataclasses.

    Returns deep-copied config instances plus the preset description.
    """
    if preset_name not in ABLATION_PRESETS:
        raise ValueError(
            f"Unknown ablation preset '{preset_name}'. Available presets: {', '.join(sorted(ABLATION_PRESETS))}"
        )

    preset = ABLATION_PRESETS[preset_name]
    model_copy = deepcopy(model_config)
    train_copy = deepcopy(train_config)

    for key, value in preset.get("model", {}).items():
        setattr(model_copy, key, value)
    for key, value in preset.get("training", {}).items():
        setattr(train_copy, key, value)

    return model_copy, train_copy, str(preset["description"])