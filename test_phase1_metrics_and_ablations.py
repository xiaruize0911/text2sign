"""Focused smoke tests for Phase 1.3 and 1.4 infrastructure."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_configs import apply_ablation_preset, list_ablation_presets
from config import ModelConfig, TrainingConfig
from validate import ModelValidator, calculate_fvd


class FakeVideoFeatureExtractor(nn.Module):
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        # Return simple per-video features without any external weights.
        return videos.mean(dim=(2, 3, 4))


class Phase13And14Tests(unittest.TestCase):
    def test_ablation_presets_cover_all_requested_variants(self):
        presets = list_ablation_presets()
        self.assertEqual(
            set(presets.keys()),
            {"frozen_clip", "no_text", "random_text", "clip_finetuned_last2"},
        )

        model_cfg = ModelConfig()
        train_cfg = TrainingConfig()

        no_text_model, _, _ = apply_ablation_preset(model_cfg, train_cfg, "no_text")
        self.assertEqual(no_text_model.text_conditioning_mode, "none")
        self.assertEqual(no_text_model.clip_trainable_layers, 0)

        random_text_model, _, _ = apply_ablation_preset(model_cfg, train_cfg, "random_text")
        self.assertEqual(random_text_model.text_conditioning_mode, "random")

        finetuned_model, _, _ = apply_ablation_preset(model_cfg, train_cfg, "clip_finetuned_last2")
        self.assertEqual(finetuned_model.text_conditioning_mode, "normal")
        self.assertEqual(finetuned_model.clip_trainable_layers, 2)

        frozen_model, _, _ = apply_ablation_preset(model_cfg, train_cfg, "frozen_clip")
        self.assertEqual(frozen_model.text_conditioning_mode, "normal")
        self.assertEqual(frozen_model.clip_trainable_layers, 0)

    def test_calculate_fvd_accepts_injected_video_backbone(self):
        real = torch.rand(4, 3, 8, 16, 16)
        fake = torch.rand(4, 3, 8, 16, 16)
        score = calculate_fvd(
            real,
            fake,
            device="cpu",
            feature_extractor=FakeVideoFeatureExtractor(),
        )
        self.assertTrue(isinstance(score, float))
        self.assertTrue(score >= 0.0)

    def test_validator_benchmark_and_backtranslation_save_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            validator = ModelValidator.__new__(ModelValidator)
            validator.results = {}
            validator.device = "cpu"
            validator.benchmark_repeats = 2
            validator.enable_backtranslation = True
            validator.artifact_dir = Path(tmp_dir)
            validator.fvd_backbone = "videomae"
            validator.pipeline = mock.Mock()
            validator.pipeline.benchmark.return_value = {
                "prompt": "Hello world",
                "repeats": 2,
                "latency_mean_sec": 1.25,
                "latency_std_sec": 0.1,
                "frames_per_second": 25.6,
                "clip_latency_sec": 1.25,
                "peak_memory_gb": None,
            }

            benchmark = ModelValidator.benchmark_inference(validator)
            self.assertIn("device", benchmark)
            benchmark_path = Path(tmp_dir) / "benchmark_results.json"
            self.assertTrue(benchmark_path.exists())

            validator._glofe_paths = mock.Mock(return_value={"root": Path(tmp_dir)})
            validator._get_fresh_dataloader = mock.Mock(return_value=[{"text": ["hello world", "thank you"]}])
            validator._run_glofe_translation = mock.Mock(side_effect=["hello world", "thanks"])
            validator._save_video_file = mock.Mock()
            validator.pipeline = mock.Mock(return_value=torch.rand(1, 3, 4, 8, 8))

            backtranslation = ModelValidator.evaluate_backtranslation(validator, max_samples=2)
            self.assertIn("bleu", backtranslation)
            backtranslation_path = Path(tmp_dir) / "backtranslation_results.json"
            self.assertTrue(backtranslation_path.exists())

            with open(backtranslation_path) as f:
                saved = json.load(f)
            self.assertEqual(saved["num_samples"], 2)


if __name__ == "__main__":
    unittest.main()