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

## Small semantic-faithfulness audit

Source: `text_to_sign/semantic_faithfulness_audit_20260320/semantic_faithfulness_results.json`

This audit applies the same external GloFE back-translation proxy to four 8-sample settings:

| Mode | Samples | BLEU | Token F1 | Sequence similarity |
|---|---:|---:|---:|---:|
| Real clips | 8 | 0.0 | 0.0 | 0.2227 |
| Intended prompt | 8 | 0.0 | 0.0 | 0.2530 |
| No text | 8 | 0.0 | 0.0 | 0.1920 |
| Random prompt | 8 | 0.0 | 0.0 | 0.2585 |

Interpretation:

- The proxy does **not** rank real clips above generated clips, so it is not a reliable absolute semantic-faithfulness metric in the present low-resolution regime.
- Removing text lowers sequence similarity relative to the intended-prompt condition (`0.1920` vs. `0.2530`), which is directionally consistent with the conditioning-control audit.
- Random-prompt generations remain close to intended-prompt generations under this proxy, reinforcing the conclusion that prompt-specific semantic control is still weak.
- Accordingly, this audit is used in the paper as a reviewer-facing negative result and limitation, not as a success metric.

## Held-out conditional-loss audit on real clips

Source: `text_to_sign/conditional_loss_audit_20260321/high_t_short.json`

This audit evaluates the strongest checkpoint directly on 16 held-out validation clips with unique prompts of at most five words. For each clip, denoising loss is measured at late diffusion timesteps ($t\in[700,1000)$) under the intended prompt, no text, and a deranged shuffled prompt, then aggregated with bootstrap confidence intervals.

| Metric | Mean | 95% bootstrap CI |
|---|---:|---:|
| Intended-prompt loss | 0.98750 | [0.98631, 0.98895] |
| No-text loss | 0.98912 | [0.98799, 0.99046] |
| Shuffled-prompt loss | 0.98755 | [0.98648, 0.98881] |
| $\Delta$(no text $-$ intended) | +0.00161 | [0.00146, 0.00177] |
| $\Delta$(shuffled $-$ intended) | +0.00005 | [-0.00019, 0.00028] |
| Prompt-ranking top-1 accuracy | 0.3125 | [0.1234, 0.5625] |
| Mean correct rank (4-way) | 2.4375 | [1.8750, 3.0000] |

Interpretation:

- This audit is more informative than the external back-translation proxy because it evaluates the model's conditional denoising score on real held-out clips rather than on generated-video translations.
- Removing text produces a consistent late-timestep loss penalty, showing that the checkpoint does use the presence of text during denoising.
- Shuffled prompts remain nearly indistinguishable from intended prompts, and prompt-ranking remains only slightly above 4-way chance, reinforcing that prompt-specific semantic control is still weak.
- Accordingly, this audit is used in the paper as a stronger reviewer-facing conditioning analysis, but still as a limitation rather than a success claim.
