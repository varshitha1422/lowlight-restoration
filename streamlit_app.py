import glob, time, cv2, numpy as np, streamlit as st
from PIL import Image, ImageOps
import lowlight_core as lc

st.set_page_config(page_title="Low-Light Restoration", layout="wide")

@st.cache_resource
def load():
    lc.load_model("zerodce.pth")
    return True
load()
MAXS = 800

st.title("Illumination-adaptive low-light restoration")
st.caption("Classical Retinex / CLAHE / guided filtering + Zero-DCE, fused by a quality-aware adaptive strategy.")

f = st.file_uploader("Upload a low-light image", type=["jpg", "jpeg", "png", "webp"])
ex = sorted(glob.glob("examples/*.png"))
img = None
if f is not None:
    try:
        img = np.array(ImageOps.exif_transpose(Image.open(f)).convert("RGB"))
    except Exception:
        st.error("Could not read this file. Please upload a valid JPG/PNG image.")
        st.stop()
elif ex:
    pick = st.selectbox("...or try an example", ["(none)"] + ex)
    if pick != "(none)":
        img = np.array(Image.open(pick).convert("RGB"))
if img is None:
    st.info("Upload an image (or pick an example) to begin.")
    st.stop()

try:
    h, w = img.shape[:2]; s = MAXS / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255
    with st.spinner("Enhancing..."):
        t = time.time(); y, inter = lc.hybrid_enhance(x); lat = time.time() - t
except Exception as e:
    st.error(f"Processing failed ({type(e).__name__}). Try another image.")
    st.stop()

a, b = lc.analyze(x), lc.analyze(y); P = lc.adaptive_params(a)
c1, c2 = st.columns(2)
c1.image(img, caption="Input", use_container_width=True)
c2.image(lc.to_u8(y), caption="Hybrid result", use_container_width=True)
st.markdown(f"**Latency:** {lat:.2f}s | **Mean brightness:** {a['mean']:.3f} -> {b['mean']:.3f} | "
            f"**Noise sigma:** {a['noise']:.4f} | **Non-uniformity:** {a['nonuni']:.2f}")
st.markdown(f"**Adaptive parameters:** gamma={P['gamma']:.2f}, CLAHE clip={P['clip']:.2f}, "
            f"bilateral sigma={P['sc']:.3f}, guided eps={P['eps']:.4f}")
st.download_button("Download result (PNG)", cv2.imencode(".png", cv2.cvtColor(lc.to_u8(y), cv2.COLOR_RGB2BGR))[1].tobytes(), "enhanced.png", "image/png")
st.subheader("Intermediate stages")
items = list(inter.items())
for r in range(0, len(items), 4):
    cols = st.columns(4)
    for col, (k, v) in zip(cols, items[r:r + 4]):
        col.image(lc.to_u8(v), caption=k, use_container_width=True)
