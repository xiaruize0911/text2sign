# Text2Sign Resubmission Runbook

This runbook records the concrete artifacts used for the IEEE Access resubmission package.

## Primary checkpoint

- Best signer-disjoint production checkpoint:
  - `text_to_sign/text_to_sign/checkpoints/text2sign_20260319_015042/best_model.pt`
- Verified best validation loss from training log:
  - `0.009989`
  - Source: `text_to_sign/step2_1_training_l4_b2_resume.log`

## Final benchmark setting

- Sampler: DDIM
- Inference steps: `8`
- Guidance scale: `5.0`
- Hardware: `NVIDIA L4`
- Final runtime artifact:
  - `text_to_sign/benchmark_results.json`

## Conditioning-control artifacts

- Summary:
  - `text_to_sign/conditioning_ablation_eval_20260320/conditioning_ablation_summary.json`
- Per-variant JSON files:
  - `.../frozen_clip/ablation_results.json`
  - `.../no_text/ablation_results.json`
  - `.../random_text/ablation_results.json`
  - `.../clip_finetuned_last2/ablation_results.json`

## Back-translation artifact

- Aggregate results:
  - `text_to_sign/backtranslation_results.json`
- Named probe file for checklist/reference convenience:
  - `text_to_sign/backtranslation_probe.json`

## Paper asset regeneration

From `text2sign_paper/`:

1. `python generate_plots.py`
2. `python update_generated_figures.py`
3. Compile `main.tex`
4. Compile `main_highlighted.tex` for the highlighted package

## Notes

- The compact audit metrics used in the manuscript are summarized in `text_to_sign/resubmission_results_summary.md`.
- The final benchmark number for the manuscript/package is the regenerated warm-start benchmark in `text_to_sign/benchmark_results.json`.
- Conditioning-control metrics are treated as coarse pixel-space probes, not linguistic correctness measures.
