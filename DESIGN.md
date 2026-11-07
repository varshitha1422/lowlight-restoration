# Design Document - P1 Low-Light Restoration (Team T01)
## 1. Problem and goals
Restore severely under-exposed, noisy, colour-distorted images without halos, over-exposure or detail loss.
## 2. Architecture
See `figs/architecture.png`. Image analysis -> adaptive parameters -> classical branch + deep branch -> quality-aware fusion -> edge-preserving post filter and highlight soft-clip.
## 3. Data
LOL: 485 train pairs (460 train / 25 validation) and 15 held-out test pairs.
## 4. Classical components
Multi-scale guided-filter illumination (max-RGB), Retinex gain with adaptive gamma, grey-world colour correction, CLAHE in LAB (adaptive clip), bilateral pre-denoising (sigma from noise estimate), guided-filter post-denoise with partial detail add-back.
## 5. Deep component
Zero-DCE DCE-Net (79,416 params, 315 KB). Stage 1: unsupervised spatial/exposure/colour/smoothness losses. Stage 2: adds L1 supervision from LOL. Best checkpoint by validation PSNR.
## 6. Hybrid strategy
Per-pixel weights = well-exposedness x illumination prior (dark -> deep, bright -> classical) + global image-quality score bias, smoothed with a guided filter to avoid seams/halos.
## 7. Evaluation
PSNR, SSIM, LPIPS, NIQE, latency, noise reduction, edge correlation, DeltaE, over-exposure. See `eval_P1_T01.md`.
## 8. Ablation, failure analysis, limitations
See evaluation file and README.
## 9. Team contributions
(Add: who built the classical branch, deep branch, fusion, app/deployment.)
