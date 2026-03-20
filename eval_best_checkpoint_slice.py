"""Compact evaluation runner for a best-checkpoint validation slice.

This avoids the very long full-suite workflow while still producing a
representative set of quantitative metrics and qualitative comparisons.
"""

import argparse
import json
import os
from pathlib import Path

from validate import ModelValidator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a compact evaluation slice for a checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", required=True, help="Path to training/validation data root")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/visual artifacts")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of validation samples to evaluate")
    parser.add_argument("--steps", type=int, default=8, help="DDIM inference steps for generated samples")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG scale for generated samples")
    parser.add_argument("--benchmark-repeats", type=int, default=2, help="Latency benchmark repeats")
    parser.add_argument("--skip-backtranslation", action="store_true", help="Skip GloFE back-translation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    validator = ModelValidator(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device="cuda",
        num_samples=args.num_samples,
        benchmark_repeats=args.benchmark_repeats,
        enable_backtranslation=not args.skip_backtranslation,
        eval_num_inference_steps=args.steps,
        eval_guidance_scale=args.guidance_scale,
    )
    validator._dataloader_params["num_workers"] = 0
    validator._dataloader_params["pin_memory"] = False
    validator._dataloader_params["persistent_workers"] = False
    validator._dataloader_params["prefetch_factor"] = None

    summary = {
        "checkpoint": args.checkpoint,
        "num_samples": args.num_samples,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "metrics": {},
    }

    summary["metrics"]["reconstruction"] = validator.evaluate_reconstruction()
    summary["metrics"]["temporal"] = validator.evaluate_temporal_consistency()
    summary["metrics"]["diversity"] = validator.evaluate_diversity()
    summary["metrics"]["benchmark"] = validator.benchmark_inference()
    summary["metrics"]["motion_realism"] = validator.evaluate_motion_realism()

    if not args.skip_backtranslation:
        summary["metrics"]["backtranslation"] = validator.evaluate_backtranslation(max_samples=min(4, args.num_samples))

    try:
        summary["metrics"]["fvd"] = validator.evaluate_fvd()
    except Exception as exc:  # pragma: no cover - best effort
        summary["metrics"]["fvd_error"] = str(exc)

    validator.compare_with_training_data(args.output_dir)

    output_path = Path(args.output_dir) / "eval_slice_results.json"
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved compact evaluation results to {output_path}")


if __name__ == "__main__":
    main()
