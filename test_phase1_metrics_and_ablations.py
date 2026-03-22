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
from torch.utils.data import WeightedRandomSampler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_configs import apply_ablation_preset, list_ablation_presets
from config import ModelConfig, TrainingConfig, apply_model_size_preset, list_model_size_presets
from dataset import SignLanguageDataset, get_dataloader
from trainer import load_state_dict_flexible
from trainer import Trainer
from utils.ema import EMA
from validate import ModelValidator, calculate_fvd


class FakeVideoFeatureExtractor(nn.Module):
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        # Return simple per-video features without any external weights.
        return videos.mean(dim=(2, 3, 4))


class Phase13And14Tests(unittest.TestCase):
    def test_semantic_v2_preset_is_available(self):
        presets = list_model_size_presets()
        self.assertIn("semantic_v2", presets)

        model_cfg = ModelConfig()
        train_cfg = TrainingConfig()
        model_cfg, train_cfg, description = apply_model_size_preset(model_cfg, train_cfg, "semantic_v2")

        self.assertEqual(model_cfg.attention_resolutions, (2, 4))
        self.assertEqual(model_cfg.clip_trainable_layers, 2)
        self.assertEqual(model_cfg.text_projection_mode, "mlp")
        self.assertGreater(train_cfg.motion_loss_weight, 0.0)
        self.assertTrue(train_cfg.use_timestep_curriculum)

    def test_training_config_has_text_discrimination_controls(self):
        train_cfg = TrainingConfig()
        self.assertTrue(hasattr(train_cfg, "text_discrimination_loss_weight"))
        self.assertTrue(hasattr(train_cfg, "text_discrimination_margin"))
        self.assertTrue(hasattr(train_cfg, "text_presence_loss_weight"))
        self.assertTrue(hasattr(train_cfg, "short_text_max_words"))
        self.assertTrue(hasattr(train_cfg, "short_text_oversample_factor"))

    def test_model_config_has_text_strength_controls(self):
        model_cfg = ModelConfig()
        self.assertTrue(hasattr(model_cfg, "use_text_global_conditioning"))
        self.assertTrue(hasattr(model_cfg, "text_time_fusion_weight"))
        self.assertTrue(hasattr(model_cfg, "use_text_feature_modulation"))
        self.assertTrue(hasattr(model_cfg, "text_feature_modulation_scale"))

    def test_short_text_oversampling_uses_weighted_sampler(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            for idx, text in enumerate(["hi", "thanks now", "this is longer text"]):
                (tmp / f"sample{idx}.gif").write_bytes(b"GIF89a")
                (tmp / f"sample{idx}.txt").write_text(text)

            with mock.patch.object(SignLanguageDataset, "_load_gif", return_value=torch.zeros(4, 3, 8, 8)):
                loader = get_dataloader(
                    data_dir=str(tmp),
                    batch_size=1,
                    image_size=8,
                    num_frames=4,
                    num_workers=0,
                    train=True,
                    split_mode="random",
                    train_ratio=1.0,
                    short_text_max_words=2,
                    short_text_oversample_factor=3.0,
                    pin_memory=False,
                )

            self.assertIsInstance(loader.sampler, WeightedRandomSampler)

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
            validator.eval_num_inference_steps = 8
            validator.eval_guidance_scale = 5.0
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

    def test_flexible_loader_accepts_shape_matched_subset(self):
        module = nn.Linear(4, 3)
        state_dict = {
            "weight": torch.randn(3, 4),
            "bias": torch.randn(3),
            "extra.weight": torch.randn(2, 2),
        }

        result = load_state_dict_flexible(module, state_dict, "linear")
        self.assertIn(result["mode"], {"strict", "partial"})
        self.assertGreaterEqual(result["loaded_keys"], 2)

    def test_load_checkpoint_weights_only_resets_training_progress(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "dummy.pt"
            torch.save(
                {
                    "model_state_dict": {"weight": torch.randn(3, 4), "bias": torch.randn(3)},
                    "text_encoder_state_dict": {"weight": torch.randn(3, 4), "bias": torch.randn(3)},
                    "optimizer_state_dict": {"state": {}, "param_groups": []},
                    "lr_scheduler_state_dict": {},
                    "global_step": 4389,
                    "epoch": 19,
                    "best_loss": 0.0099,
                },
                checkpoint_path,
            )

            trainer = Trainer.__new__(Trainer)
            trainer.base_model = nn.Linear(4, 3)
            trainer.base_text_encoder = nn.Linear(4, 3)
            trainer.model_config = mock.Mock(partial_load_on_resume=True)
            trainer.optimizer = mock.Mock()
            trainer.lr_scheduler = mock.Mock()
            trainer.scaler = None
            trainer.ema = None
            trainer.device = torch.device("cpu")
            trainer.global_step = 123
            trainer.epoch = 7
            trainer.best_loss = 1.23
            trainer._move_optimizer_state_to_device = mock.Mock()

            Trainer.load_checkpoint(trainer, str(checkpoint_path), resume_training=False)

            self.assertEqual(trainer.global_step, 0)
            self.assertEqual(trainer.epoch, 0)
            self.assertEqual(trainer.best_loss, float("inf"))
            trainer.optimizer.load_state_dict.assert_not_called()
            trainer.lr_scheduler.load_state_dict.assert_not_called()
            trainer.optimizer.zero_grad.assert_called_once_with(set_to_none=True)

    def test_ema_load_skips_shape_mismatches(self):
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
        ema = EMA(model, device=torch.device("cpu"))

        bad_state = ema.state_dict()
        bad_state["shadow"]["0.weight"] = torch.randn(99, 99)

        ema.load_state_dict(bad_state)
        self.assertEqual(ema.shadow["0.weight"].shape, model[0].weight.shape)

        # Ensure apply_shadow does not crash even if a bad tensor sneaks in later.
        ema.shadow["0.weight"] = torch.randn(99, 99)
        ema.apply_shadow()
        ema.restore()


if __name__ == "__main__":
    unittest.main()