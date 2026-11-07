1RUA24CSE0520,1RUA24CSE0473

# Evaluation - P1 (Low-light restoration) - Team T01
Test set: LOL eval15 (15 unseen pairs, never used for training or checkpoint selection). Device: cuda. Model: Zero-DCE-style DCE-Net, 79,416 params, 315 KB.

## 1. Comparison: classical vs deep learning vs hybrid
| Method | PSNR↑ | SSIM↑ | LPIPS↓ | NIQE↓ | Latency ms↓ | Noise reduction ×↑ | Edge corr. vs GT↑ | ΔE2000↓ | Over-exposed %↓ |
|---|---|---|---|---|---|---|---|---|---|
| Input (no enhancement) | 7.773 | 0.191 | 0.560 | 6.749 | 0.018 | 1.000 | 0.836 | 37.076 | 0.000 |
| Classical only | 15.642 | 0.708 | 0.298 | 5.215 | 101.629 | 3.611 | 0.851 | 15.874 | 0.000 |
| Deep learning only (Zero-DCE) | 17.834 | 0.542 | 0.395 | 8.069 | 21.135 | 1.061 | 0.852 | 14.702 | 0.002 |
| Hybrid (proposed) | 17.464 | 0.770 | 0.269 | 5.614 | 281.949 | 6.438 | 0.886 | 13.569 | 0.000 |

## 2. Ablation study
| Method | PSNR↑ | SSIM↑ | LPIPS↓ | NIQE↓ | Latency ms↓ | Noise reduction ×↑ | Edge corr. vs GT↑ | ΔE2000↓ | Over-exposed %↓ |
|---|---|---|---|---|---|---|---|---|---|
| Full hybrid | 17.464 | 0.770 | 0.269 | 5.614 | 257.427 | 6.438 | 0.886 | 13.569 | 0.000 |
| Hybrid - no pre-denoise | 16.761 | 0.695 | 0.274 | 6.110 | 253.952 | 1.896 | 0.891 | 14.546 | 0.000 |
| Hybrid - no post edge-preserving filter | 17.466 | 0.767 | 0.263 | 5.396 | 241.809 | 5.571 | 0.889 | 13.583 | 0.000 |
| Hybrid - no illumination prior in fusion | 17.807 | 0.773 | 0.267 | 5.642 | 265.063 | 6.499 | 0.886 | 13.165 | 0.000 |
| Hybrid - no quality-aware weighting | 17.428 | 0.771 | 0.270 | 5.591 | 267.298 | 5.966 | 0.883 | 13.251 | 0.000 |
| Hybrid - no fusion (deep + post filter only) | 18.842 | 0.765 | 0.272 | 5.786 | 178.140 | 7.389 | 0.881 | 12.745 | 0.000 |
| Classical - full | 15.642 | 0.708 | 0.298 | 5.215 | 101.953 | 3.611 | 0.851 | 15.874 | 0.000 |
| Classical - no Retinex (gamma only) | 15.906 | 0.741 | 0.312 | 5.501 | 93.747 | 5.887 | 0.870 | 16.311 | 0.000 |
| Classical - no CLAHE | 14.398 | 0.686 | 0.289 | 6.314 | 95.755 | 5.156 | 0.868 | 19.816 | 0.000 |
| Classical - no denoising | 15.271 | 0.480 | 0.462 | 8.502 | 86.055 | 0.692 | 0.822 | 16.634 | 0.000 |
| Classical - no colour correction | 15.778 | 0.708 | 0.298 | 5.190 | 112.239 | 3.625 | 0.851 | 15.659 | 0.008 |
| Classical - no post filter | 15.589 | 0.695 | 0.297 | 5.142 | 78.584 | 3.170 | 0.852 | 15.907 | 0.004 |

## 3. Generalisation across illumination (synthetic darkening + noise, PSNR / SSIM)
| Brightness scale | Input (no enhancement) | Classical only | Deep learning only (Zero-DCE) | Hybrid (proposed) |
|---|---|---|---|---|
| 0.3 | 13.67 dB / 0.546 | 18.43 dB / 0.788 | 11.14 dB / 0.445 | 16.70 dB / 0.746 |
| 0.1 | 9.95 dB / 0.297 | 17.05 dB / 0.685 | 11.96 dB / 0.291 | 15.23 dB / 0.669 |
| 0.03 | 8.16 dB / 0.131 | 15.65 dB / 0.504 | 12.12 dB / 0.109 | 14.38 dB / 0.554 |

Figures: `figs/comparison.png`, `figs/intermediates.png`, `figs/robustness.png`, `figs/failures.png`.

## 4. Failure analysis
- **Test image #5** - PSNR 12.94, SSIM 0.718, DeltaE 20.2, over-exposed 0.00%. Likely cause (auto-heuristic, verify visually): brightness/colour mismatch with the reference (global exposure ambiguity) and softened texture.
- **Test image #8** - PSNR 13.08, SSIM 0.732, DeltaE 19.2, over-exposed 0.00%. Likely cause (auto-heuristic, verify visually): brightness/colour mismatch with the reference (global exposure ambiguity) and softened texture.
- **Test image #3** - PSNR 13.63, SSIM 0.655, DeltaE 19.7, over-exposed 0.00%. Likely cause (auto-heuristic, verify visually): brightness/colour mismatch with the reference (global exposure ambiguity) and softened texture.

Extreme darkness (scale 0.03) degrades every method: noise dominates the signal, so colour and texture cannot be recovered reliably.
