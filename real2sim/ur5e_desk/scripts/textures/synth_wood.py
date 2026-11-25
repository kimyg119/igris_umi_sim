"""Full desk-top wood albedo (1400 x 600 mm) from the rectified ~0.3 x 0.5 m patch.
The patch is kept as-is (natural figure), stretched 1.55x along the grain and mirror-tiled 3x; the kink that a
mirror puts into slanted grain lines is rounded off with a per-row warp so it reads as a natural undulation.
The single knot is kept in the middle tile only."""
import numpy as np, cv2
MM_PX = 0.30; OW, OH = int(1400 / MM_PX), int(600 / MM_PX)
SRC_W, TILES, KINK_W = 1000, 3, 140          # px of source used per row, mirror tiles across the desk, kink rounding width (px)
KNOT = (640, 908, 125, 62, -330)             # cx, cy, half-axes (px) of the knot in the exemplar, x-offset of the clean donor
E = np.load("exemplar_lin.npy").astype(np.float32); M = np.load("exemplar_mask.npy")
H, W = E.shape[:2]
yy, xx = np.mgrid[:H, :W].astype(np.float32)
P = np.stack([np.ones_like(xx), xx / W, yy / H, (xx / W) ** 2, (yy / H) ** 2, xx * yy / W / H], -1)   # de-light: smooth falloff only
coef = np.linalg.lstsq(P[M][::7], np.log(np.maximum(E.mean(2), 1e-4))[M][::7], rcond=None)[0]
E = E / np.exp(P @ coef - coef[0])[..., None]
kx, ky, ka, kb, kdx = KNOT
kmask = cv2.GaussianBlur((((xx - kx) / ka) ** 2 + ((yy - ky) / kb) ** 2 < 1).astype(np.float32), (0, 0), 14)[..., None]
donor = np.roll(E, -kdx, axis=1); ring = (kmask[..., 0] > 0.02) & (kmask[..., 0] < 0.3)
E_clean = E * (1 - kmask) + donor * (E[ring].mean(0) / donor[kmask[..., 0] > 0.5].mean(0)) * kmask
Me = cv2.erode(M.astype(np.uint8), np.ones((1, 25), np.uint8)) > 0
xl = np.zeros(H, np.float32); xr = np.zeros(H, np.float32)
for y in range(H):
    r = np.flatnonzero(Me[y])
    if len(r):
        best = max(np.split(r, np.flatnonzero(np.diff(r) > 1) + 1), key=len); xl[y], xr[y] = best[0], best[-1]
rows = np.flatnonzero(xr - xl > 600); y0, y1 = rows[0], rows[-1]
sm = lambda v: cv2.GaussianBlur(v.reshape(-1, 1), (1, 0), 25).ravel()
xl, xr = sm(xl) + 4, np.minimum(sm(xr) - 4, sm(xl) + 4 + SRC_W)      # rows narrower than SRC_W get stretched a bit more
TW = OW / TILES
ys = (y0 + (np.arange(OH) + 0.5) / OH * (y1 - y0)).astype(np.float32)
l, r = np.interp(ys, np.arange(H), xl), np.interp(ys, np.arange(H), xr)
print(f"rows {y0}..{y1}: y-stretch {OH / (y1 - y0):.2f}, x-stretch {TW / (r - l).max():.2f}..{TW / (r - l).min():.2f}, tile {TW * MM_PX:.0f} mm")
xo = np.arange(OW) + 0.5; t = np.floor(xo / TW).astype(int); f = xo / TW - t
f = np.where(t % 2 == 1, 1 - f, f)                                     # odd tiles are mirrored
XS = (l[:, None] + f[None] * (r - l)[:, None]).astype(np.float32); YS = np.repeat(ys[:, None], OW, 1)
tiled = np.where((t == TILES // 2)[None, :, None], cv2.remap(E, XS, YS, cv2.INTER_CUBIC), cv2.remap(E_clean, XS, YS, cv2.INTER_CUBIC))
# grain slope dy/dx at each mirror line (structure tensor on the tiled image, just left of the line)
g = cv2.GaussianBlur(tiled.mean(2), (0, 0), 2.0); gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
disp = np.zeros((OH, OW), np.float32)
for k in range(1, TILES):
    c = int(round(k * TW)); sl = slice(c - 70, c - 10)
    jxx, jxy, jyy = [cv2.GaussianBlur((a[:, sl] * b[:, sl]).mean(1).reshape(-1, 1), (1, 0), 12).ravel() for a, b in ((gx, gx), (gx, gy), (gy, gy))]
    slope = -jxy / np.maximum(jyy, 1e-9)                               # iso-line slope for near-horizontal grain
    slope = np.clip(slope, -0.15, 0.15); tt = np.arange(OW) - c
    disp += slope[:, None] * (np.abs(tt) - np.sqrt(tt ** 2 + KINK_W ** 2) + KINK_W)[None]
    print(f"mirror line at {c * MM_PX:.0f} mm: |slope| mean {np.abs(slope).mean():.4f} max {np.abs(slope).max():.3f}")
out = cv2.remap(tiled, np.repeat(xo[None].astype(np.float32) - 0.5, OH, 0), np.repeat(np.arange(OH, dtype=np.float32)[:, None], OW, 1) - disp, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
# pores right of each mirror line come from elsewhere on the same rows, so the fine detail is not a mirror image
lo = cv2.GaussianBlur(out, (0, 0), 5); hp = out / np.maximum(lo, 1e-4)
for k in range(1, TILES):
    c = int(round(k * TW)); w = np.clip(1 - np.abs(np.arange(OW) - (c + 170)) / 170.0, 0, 1) ** 0.5
    out = out * (1 - w[None, :, None]) + lo * np.roll(hp, -int(TW * 0.37), axis=1) * w[None, :, None]
np.save("desk_albedo_rel.npy", out)
from colorlin import srgb
prev = (srgb(out / np.percentile(out, 99.5) * 0.9) * 255).astype(np.uint8)[..., ::-1]
cv2.imwrite("desk_synth_preview.jpg", cv2.resize(prev, None, fx=0.33, fy=0.33, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])

