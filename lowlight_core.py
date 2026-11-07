"""Hybrid classical + Zero-DCE low-light enhancement (shared by training/eval and the web app)."""
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
F32 = np.float32

def to_u8(x): return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
def gray(x): return (x[..., 0]*0.299 + x[..., 1]*0.587 + x[..., 2]*0.114).astype(F32)
def box(x, r): return cv2.boxFilter(x, -1, (2*r+1, 2*r+1), borderType=cv2.BORDER_REFLECT)
def as3(m):
    m = np.asarray(m, F32)
    return np.repeat(m[..., None], 3, 2) if m.ndim == 2 else m

def guided_filter(I, p, r=8, eps=1e-3):
    I = I.astype(F32)
    if p.ndim == 3:
        return np.stack([guided_filter(I, p[..., c], r, eps) for c in range(p.shape[2])], -1)
    p = p.astype(F32)
    mI, mp = box(I, r), box(p, r)
    a = (box(I*p, r) - mI*mp) / (box(I*I, r) - mI*mI + eps)
    b = mp - a*mI
    return (box(a, r)*I + box(b, r)).astype(F32)

def estimate_noise(g):  # Immerkaer fast noise sigma estimate (0..1 units)
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], F32)
    c = cv2.filter2D(g.astype(F32), -1, M)[1:-1, 1:-1]
    H, W = g.shape
    return float(np.abs(c).sum() * np.sqrt(0.5*np.pi) / (6.0*(W-2)*(H-2)))

def analyze(img):
    g = gray(img)
    return dict(mean=float(g.mean()), std=float(g.std()), noise=estimate_noise(g),
                nonuni=float(np.std(cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA)) / (g.mean()+1e-4)))

def adaptive_params(st):
    m, n = st["mean"], st["noise"]
    return dict(gamma=float(np.clip(0.32+1.4*m, 0.35, 0.80)),      # darker -> stronger lift
                sc=float(np.clip(0.03+8*n, 0.04, 0.20)),            # bilateral colour sigma
                clip=float(np.clip(3.5-40*n, 1.2, 3.0)),            # CLAHE clip limit
                eps=float(np.clip(0.003+0.4*n, 0.003, 0.012)),      # guided-filter eps (post)
                keep=float(np.clip(0.8-10*n, 0.4, 0.8)))            # detail kept after post filter

def soft_clip(x, k=0.85):  # highlight roll-off -> avoids over-exposure
    return np.where(x <= k, x, k + (1-k)*np.tanh((x-k)/(1-k))).astype(F32)

def gray_world(y, s=0.5):
    m = y.reshape(-1, 3).mean(0) + 1e-6
    return np.clip(y * np.clip((m.mean()/m)**s, 0.8, 1.25), 0, 1).astype(F32)

def post_denoise(y, P):  # edge-preserving: guided filter base + partial detail add-back
    base = guided_filter(gray(y), y, r=4, eps=P["eps"])
    return np.clip(base + P["keep"]*(y-base), 0, 1).astype(F32)

# ---------------- classical branch ----------------
def classical_enhance(img, st=None, use_denoise=True, use_retinex=True, use_clahe=True,
                      use_color=True, use_post=True):
    img = img.astype(F32); st = st or analyze(img); P = adaptive_params(st); I = {}
    x = cv2.bilateralFilter(img, 7, P["sc"], 5) if use_denoise else img.copy()
    mx = x.max(2)
    if use_retinex:
        r0 = max(4, min(img.shape[:2])//50)
        L = np.mean([guided_filter(mx, mx, r, 1e-2) for r in (r0, 3*r0, 8*r0)], 0).astype(F32)  # multi-scale illumination
        L = np.clip(L, 0.02, 1.0)
        gain = np.minimum(L**P["gamma"] / L, 12.0)                     # R*L^gamma  (= x * gain)
        y = x * gain[..., None]
        I["illum"] = L; I["reflect"] = np.clip(x / L[..., None], 0, 1)
    else:
        y = x ** P["gamma"]; I["illum"] = mx; I["reflect"] = x
    y = soft_clip(y)
    if use_color: y = gray_world(y)
    if use_clahe:
        lab = cv2.cvtColor(to_u8(y), cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(P["clip"], (8, 8)).apply(lab[..., 0])
        y = 0.6*(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(F32)/255) + 0.4*y
    if use_post: y = post_denoise(y, P)
    I["denoised"] = x
    return np.clip(y, 0, 1).astype(F32), I

# ---------------- deep branch: Zero-DCE ----------------
class DCENet(nn.Module):
    def __init__(s, c=32):
        super().__init__()
        s.c1 = nn.Conv2d(3, c, 3, 1, 1); s.c2 = nn.Conv2d(c, c, 3, 1, 1)
        s.c3 = nn.Conv2d(c, c, 3, 1, 1); s.c4 = nn.Conv2d(c, c, 3, 1, 1)
        s.c5 = nn.Conv2d(2*c, c, 3, 1, 1); s.c6 = nn.Conv2d(2*c, c, 3, 1, 1)
        s.c7 = nn.Conv2d(2*c, 24, 3, 1, 1)
    def forward(s, x):
        r = F.relu
        x1 = r(s.c1(x)); x2 = r(s.c2(x1)); x3 = r(s.c3(x2)); x4 = r(s.c4(x3))
        x5 = r(s.c5(torch.cat([x3, x4], 1))); x6 = r(s.c6(torch.cat([x2, x5], 1)))
        a = torch.tanh(s.c7(torch.cat([x1, x6], 1)))
        e = x
        for ri in torch.split(a, 3, 1):
            e = e + ri*(e*e - e)          # light-enhancement curve, 8 iterations
        return e, a

_model = None
def load_model(path="zerodce.pth"):
    global _model
    m = DCENet().to(DEV); m.load_state_dict(torch.load(path, map_location=DEV)); m.eval(); _model = m
    return m

@torch.no_grad()
def dl_enhance(img, model=None):
    model = model or _model
    t = torch.from_numpy(np.ascontiguousarray(img.astype(F32))).permute(2, 0, 1)[None].to(DEV)
    e, _ = model(t)
    return e[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy().astype(F32)

# ---------------- quality-aware hybrid ----------------
def well_exposed(x, s=0.25):
    return (np.exp(-((x.mean(2)-0.5)**2)/(2*s*s)) * (0.2 + x.std(2))).astype(F32)

def quality_score(x):  # simple no-reference quality proxy: exposure, contrast, clipping, noise
    l = gray(x)
    expo = np.exp(-((l.mean()-0.45)**2)/(2*0.15**2))
    return float(expo + 0.5*min(l.std()/0.25, 1) - 3*(l > 0.97).mean() - 1.0*(l < 0.05).mean()
                 - 5*estimate_noise(l)/(l.mean()+1e-3))

def hybrid_enhance(img, model=None, use_fusion=True, use_illum_prior=True, use_predenoise=True,
                   use_quality=True, use_post=True):
    img = img.astype(F32); st = analyze(img); P = adaptive_params(st)
    cl, ci = classical_enhance(img, st)
    x = cv2.bilateralFilter(img, 7, P["sc"], 5) if use_predenoise else img
    dl = dl_enhance(x, model)
    L = ci["illum"]; Ln = np.clip(L / (np.percentile(L, 99)+1e-6), 0, 1)
    if use_fusion:
        beta = float(np.clip(st["nonuni"], 0.3, 1.5))     # non-uniform lighting -> trust illumination prior more
        pd_ = (0.5 + (1-Ln)*beta) if use_illum_prior else 1.0   # dark regions -> favour deep curves
        pc_ = (0.5 + Ln*beta) if use_illum_prior else 1.0       # bright regions -> favour classical
        w = well_exposed(dl)*pd_ / (well_exposed(dl)*pd_ + well_exposed(cl)*pc_ + 1e-6)
        if use_quality:                                    # image-quality-aware global bias
            g = 1/(1+np.exp(-8*(quality_score(dl)-quality_score(cl))))
            w = np.clip(w + 0.6*(g-0.5), 0, 1)
        w = np.clip(guided_filter(gray(cl), w.astype(F32), r=15, eps=1e-2), 0, 1)  # smooth, no seams/halos
        y = w[..., None]*dl + (1-w[..., None])*cl
    else:
        w = np.ones(img.shape[:2], F32); y = dl
    if use_post: y = post_denoise(y, P)
    y = np.clip(gray_world(soft_clip(y), 0.3), 0, 1).astype(F32)
    inter = {"1 Input": img, "2 Illumination map (classical)": as3(L/L.max()),
             "3 Reflectance (Retinex)": ci["reflect"], "4 Pre-denoised": x,
             "5 Classical output": cl, "6 Deep (Zero-DCE) output": dl,
             "7 Fusion map (white = deep)": as3(w), "8 Hybrid output": y}
    return y, inter
