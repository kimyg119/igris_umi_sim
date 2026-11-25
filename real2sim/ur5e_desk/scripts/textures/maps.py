"""PBR maps for the desk top and the wall."""
import numpy as np, cv2
from colorlin import srgb, hlg_to_linear709
import os
OUT = os.environ["TEX_OUT"]
DESK_ALBEDO = np.array([0.404, 0.270, 0.155])   # linear; chroma = desk/wall radiance ratio in one frame x white-paint albedo
WALL_ALBEDO = np.array([0.86, 0.855, 0.83])
MM_PX = 0.30
rng = np.random.default_rng(1)
def save_srgb(name, lin): cv2.imwrite(f"{OUT}/{name}", (srgb(lin) * 255 + 0.5).astype(np.uint8)[..., ::-1])
def normal_map(height_px):                       # height in pixel units -> tangent-space normal (OpenGL convention, +Y up)
    gx = cv2.Sobel(height_px, cv2.CV_32F, 1, 0, ksize=3) / 8; gy = cv2.Sobel(height_px, cv2.CV_32F, 0, 1, ksize=3) / 8
    n = np.dstack([-gx, gy, np.ones_like(gx)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
    return ((n * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)
# ---- desk
a = np.load("desk_albedo_rel.npy"); a = a * (DESK_ALBEDO / a.mean((0, 1)))
save_srgb("desk_wood_albedo.png", a)
L = a.mean(2); hp = L / cv2.GaussianBlur(L, (0, 0), 4) - 1                      # pores are darker than their surround
h = cv2.GaussianBlur(np.clip(hp / 0.10, -1, 1), (0, 0), 0.7) * 0.15             # ~45 um emboss at 0.3 mm/px
cv2.imwrite(f"{OUT}/desk_wood_normal.png", normal_map(h.astype(np.float32))[..., ::-1])
cv2.imwrite(f"{OUT}/desk_wood_roughness.png", (np.clip(0.47 - hp * 0.8, 0.36, 0.66) * 255).astype(np.uint8))
print("desk albedo mean (lin)", a.mean((0, 1)).round(3), "contrast p1/p99 of luminance", np.percentile(L, [1, 99]).round(3))
# ---- wall stipple: isotropic random-phase noise with the measured radial spectrum of the paint texture
A = hlg_to_linear709("lin/n_0144.png").mean(2); p = A[700:1212, 240:752]
p = p / cv2.GaussianBlur(p, (0, 0), 12) - 1
win = np.outer(np.hanning(512), np.hanning(512)); F = np.abs(np.fft.fft2(p * win))
fy, fx = np.meshgrid(np.fft.fftfreq(512), np.fft.fftfreq(512), indexing="ij"); fr = np.hypot(fx, fy)
bins = np.linspace(0, 0.5, 129); idx = np.digitize(fr.ravel(), bins); prof = np.array([F.ravel()[idx == i].mean() if (idx == i).any() else 0 for i in range(1, 130)])
N = 2048; FY, FX = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing="ij"); FR = np.hypot(FX, FY)
amp = np.interp(FR, bins, prof); amp[FR < 0.012] = 0; amp[FR > 0.33] = 0     # keep the stipple band, drop blotches and codec noise
st = np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((N, N))))); st /= st.std()
peak = bins[np.argmax(prof * (bins > 0.02) * (bins < 0.33))]
print(f"wall stipple dominant wavelength {MM_PX / max(peak, 1e-6):.1f} mm; observed shading contrast {p.std() * 100:.1f} %")
g = np.hypot(*np.gradient(st)).std(); hgt = (st * (0.045 / g)).astype(np.float32)   # RMS slope ~0.045 (2.6 deg)
cv2.imwrite(f"{OUT}/wall_stipple_normal.png", normal_map(hgt)[..., ::-1])
# ---- wall albedo: white paint + faint mottling + the real scuff marks from the video
W = 4096; WALL_MM = 3000                                                      # texture covers 3 x 3 m of wall
fy_, fx_ = np.meshgrid(np.fft.fftfreq(W), np.fft.fftfreq(W), indexing="ij")
mott = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((W, W))) * np.exp(-2 * (np.pi * 90) ** 2 * (fx_ ** 2 + fy_ ** 2)))).astype(np.float32); mott /= mott.std()   # periodic
alb = np.ones((W, W, 3), np.float32) * WALL_ALBEDO * (1 + 0.012 * mott[..., None])
def scuff(frame, box, cx, cy, mm_px=0.33):
    y0, y1, x0, x1 = box; Lw = hlg_to_linear709(f"lin/n_{frame:04d}.png").mean(2)
    r = (Lw / cv2.GaussianBlur(Lw, (0, 0), 45))[y0:y1, x0:x1]; r = 1 - 0.6 * (1 - np.clip((r - 1) / 0.985 + 1 + 0.015, 0.55, 1.0))   # darkening only, ignore <1.5 %, softened
    r = cv2.resize(r.astype(np.float32), None, fx=mm_px / (WALL_MM / W), fy=mm_px / (WALL_MM / W), interpolation=cv2.INTER_AREA)
    f = np.outer(np.hanning(r.shape[0]) ** 0.4, np.hanning(r.shape[1]) ** 0.4); r = 1 - (1 - r) * f
    alb[cy:cy + r.shape[0], cx:cx + r.shape[1]] *= r[..., None]
scuff(102, (0, 760, 0, 520), 2560, 2420)      # smudge + three dots: x ~ +0.45 m, ~0.4 m above the desk top
scuff(120, (0, 640, 0, 300), 760, 1950)       # x ~ -0.9 m, z ~ 1.5 m
save_srgb("wall_paint_albedo.png", alb)
print("wall albedo mean", alb.mean((0, 1)).round(3), "min", alb.min().round(3))

# ---- floor: speckled beige vinyl tile (as in the basket photos), tileable 0.6 m
rng = np.random.default_rng(5); N = 1024
def noise(sig):
    fy, fx = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing="ij")
    n = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((N, N))) * np.exp(-2 * (np.pi * sig) ** 2 * (fx ** 2 + fy ** 2)))); return n / n.std()
spk = np.clip(noise(1.2) * 0.6 + noise(4) * 0.4, -2.5, 2.5)
save_srgb("floor_vinyl_albedo.png", np.array([0.56, 0.51, 0.41]) * (1 + 0.07 * spk[..., None]) * (1 + 0.03 * noise(40)[..., None]))
