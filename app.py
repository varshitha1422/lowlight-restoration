import glob, time, cv2, numpy as np, gradio as gr
import lowlight_core as lc
lc.load_model("zerodce.pth")
MAXS = 800

def run(img):
    if img is None: raise gr.Error("Please upload or pick an image first.")
    try:
        img = np.asarray(img)
        img = np.repeat(img[..., None], 3, 2) if img.ndim == 2 else img[..., :3]
        h, w = img.shape[:2]; s = MAXS / max(h, w)
        if s < 1: img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32) / 255
        t = time.time(); y, inter = lc.hybrid_enhance(x); lat = time.time() - t
        a, b = lc.analyze(x), lc.analyze(y); P = lc.adaptive_params(a)
        md = (f"**Latency:** {lat:.2f}s | **Mean brightness:** {a['mean']:.3f} -> {b['mean']:.3f} | "
              f"**Noise sigma:** {a['noise']:.4f} | **Non-uniformity:** {a['nonuni']:.2f}\n\n"
              f"**Adaptive parameters chosen:** gamma={P['gamma']:.2f}, CLAHE clip={P['clip']:.2f}, "
              f"bilateral sigma={P['sc']:.3f}, guided eps={P['eps']:.4f}")
        return lc.to_u8(y), [(lc.to_u8(v), k) for k, v in inter.items()], md
    except Exception as e:
        raise gr.Error(f"Could not process this image ({type(e).__name__}). Try a standard JPG/PNG.")

with gr.Blocks(title="Low-Light Image Restoration") as demo:
    gr.Markdown("# Illumination-adaptive low-light restoration\nClassical Retinex/CLAHE/guided filtering + Zero-DCE, fused with a quality-aware adaptive strategy.")
    with gr.Row():
        inp = gr.Image(type="numpy", label="Low-light image")
        out = gr.Image(label="Hybrid result")
    btn = gr.Button("Enhance", variant="primary")
    stats = gr.Markdown()
    gal = gr.Gallery(label="Intermediate stages", columns=4)
    btn.click(run, inp, [out, gal, stats])
    ex = sorted(glob.glob("examples/*.png"))
    if ex: gr.Examples(ex, inp)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
