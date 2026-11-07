# P1 - Robust Low-Light Image Restoration (Team T01)
Hybrid of classical CV (multi-scale guided illumination, Retinex, CLAHE, bilateral/guided denoising, colour correction, adaptive parameters) and Zero-DCE (fine-tuned on LOL), fused by a quality-aware adaptive weight map.

**Run:** `pip install -r requirements.txt && streamlit run streamlit_app.py`  (or `docker build -t p1 . && docker run -p 7860:7860 p1`)

**Reproduce results:** run the training/evaluation notebook (`notebook.ipynb`, or the code in `lowlight_core.py` + notebook cell) on Colab T4 (~15 min). Seeds fixed to 0.

| Method | PSNR | SSIM | LPIPS | NIQE | Latency ms |
|---|---|---|---|---|---|
| Classical | 15.64 | 0.708 | 0.298 | 5.21 | 102 |
| Deep (Zero-DCE) | 17.83 | 0.542 | 0.395 | 8.07 | 21 |
| Hybrid | 17.46 | 0.770 | 0.269 | 5.61 | 282 |

Full tables, ablation and failure analysis: `eval_P1_T01.md`. Architecture: `figs/architecture.png`.

## Limitations and ethics
Trained on LOL (mostly indoor, 400x600) so outdoor/night scenes may be less accurate; extremely dark inputs are dominated by noise; enhancement can reveal content that was not visible, so do not use it as forensic evidence; images are processed in memory and not stored.
