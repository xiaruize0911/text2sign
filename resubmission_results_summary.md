# Text2Sign Resubmission Results Summary

This file consolidates the final numbers cited in the resubmission package and records their source artifacts.

## Main signer-disjoint production run

- Run ID: `text2sign_20260319_015042`
- Best validation loss: `0.009989`
- Source: `text_to_sign/step2_1_training_l4_b2_resume.log`

## Compact checkpoint audit used in manuscript text

These values come from the verified compact audit used during the revision cycle.

- SSIM: `0.2403 ± 0.0238`
- PSNR: `15.11 ± 0.42 dB`
- Temporal consistency: `1.0000 ± 0.0000`
- Motion magnitude: `0.0987 ± 0.0072`

These should be interpreted as coarse reconstruction/motion proxies rather than sign-linguistic correctness metrics.

## Final runtime benchmark (artifact-backed)

Source: `text_to_sign/benchmark_results.json`

- Prompt: `Hello world`
- Repeats: `5`
- Inference steps: `8`
- Guidance scale: `5.0`
- Mean latency: `12.596996582399697 s`
- Latency std: `1.3947133408013805 s`
- Effective throughput: `2.54028805919578 FPS`
- Peak GPU memory: `3.1238250732421875 GB`
- Device: `NVIDIA L4`

Rounded manuscript values:

- `12.60 s / clip`
- `2.54 FPS`
- `3.12 GB peak`

## Conditioning-control audit

Source: `text_to_sign/conditioning_ablation_eval_20260320/conditioning_ablation_summary.json`

| Variant | Cross-prompt dist. | Same-prompt dist. | Temporal consistency | Motion magnitude |
|---|---:|---:|---:|---:|
| Frozen CLIP | 0.015254 | 0.015600 | 0.999998 | 0.094454 |
| No text | 0.020949 | 0.021161 | 0.999958 | 0.108000 |
| Random text | 0.015257 | 0.015600 | 0.999998 | 0.094621 |
| CLIP fine-tuned (last 2) | 0.015266 | 0.015611 | 0.999998 | 0.094495 |

Interpretation:

- Removing text increases same-prompt drift and motion magnitude.
- Random-text and partially fine-tuned CLIP remain very close to the frozen-CLIP baseline on these coarse proxies.
- This is treated as a negative result indicating that the current checkpoint still uses text only weakly.

## Back-translation artifact

Source: `text_to_sign/backtranslation_results.json`

- Samples: `4`
- BLEU: `0.0`
- Token F1: `0.0`
- Exact match: `0.0`

The back-translation results remain extremely weak and are therefore discussed only as a limitation/proxy rather than a success metric.
