"""Rectify sharp desk frames to a metric top-down wood exemplar (scene-linear BT.709)."""
import numpy as np, cv2, glob, sys
from colorlin import hlg_to_linear709, srgb
F_PX = 1500.0          # iPhone 17 main cam, 1080p video (26 mm equiv, ~stabilisation crop) -- weakly affects result
CAM_H = 0.36           # m, camera height above desk. NOT observable from the video: assumes the cubes are 25 mm.
MM_PX = 0.30           # output resolution
REF, OTHER = 2, 343
SET = [0, 1, 2, 3, 4, 338, 339, 340, 341, 342, 343, 344, 345, 346]
K = np.array([[F_PX, 0, 540], [0, F_PX, 960], [0, 0, 1]])
lin = {i: hlg_to_linear709(f"lin/n_{i:04d}.png") for i in SET}
def g8(x): return (srgb(x / np.percentile(x, 99.5) * 0.9).mean(2) * 255).astype(np.uint8)
sift = cv2.SIFT_create(8000, contrastThreshold=0.01)
feat = {i: sift.detectAndCompute(g8(lin[i]), None) for i in SET}
def homog(i, j):
    (ka, da), (kb, db) = feat[i], feat[j]
    m = [x for x, y in cv2.BFMatcher().knnMatch(da, db, k=2) if x.distance < 0.75 * y.distance]
    pa = np.float32([ka[x.queryIdx].pt for x in m]); pb = np.float32([kb[x.trainIdx].pt for x in m])
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 1.5)
    return H, int(mask.sum())
# plane normal in REF camera coords
H, n_in = homog(REF, OTHER)
_, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
n = min((N.ravel() for N in Ns if N[2] < 0 and N[1] < 0), key=lambda v: abs(v[0]))
print("plane normal (ref cam):", n.round(4), "tilt from normal:", np.degrees(np.arccos(-n[2])).round(1), "deg, inliers", n_in)
# desk/wall boundary in REF: wood is saturated, wall is not
A = lin[REF]; sat = (A.max(2) - A.min(2)) / (A.max(2) + 1e-6)
wood = cv2.morphologyEx((sat > 0.35).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
xs = np.arange(40, 1040, 8); ys = np.array([np.argmax(wood[:, x] > 0) for x in xs])
ok = np.abs(ys - np.poly1d(np.polyfit(xs, ys, 1))(xs)) < 15
a, b = np.polyfit(xs[ok], ys[ok], 1)
print("junction line y = %.4f x + %.1f (%d pts)" % (a, b, ok.sum()))
# in-plane x axis = junction direction: back-project two junction points onto the plane
def ray_hit(u, v):
    d = np.linalg.inv(K) @ np.array([u, v, 1.0]); return d * (-CAM_H / (n @ d))
p0, p1 = ray_hit(100, a * 100 + b), ray_hit(900, a * 900 + b)
ex = (p1 - p0) / np.linalg.norm(p1 - p0); ez = -n; ey = np.cross(ez, ex)
R = np.stack([ex, ey, ez])                       # ref cam -> plane frame (x along wall, y away from wall, z down)
f_out = CAM_H / (MM_PX * 1e-3)
wall_y = (R @ p0)[1]                              # plane-frame y of the junction
corners = np.array([[0, 0, 1], [1080, 0, 1], [1080, 1920, 1], [0, 1920, 1]], float).T
c = R @ np.linalg.inv(K) @ corners; c = c[:2] / c[2] * f_out
x0, x1, y1 = c[0].min(), c[0].max(), c[1].max(); y0 = wall_y / (R @ p0)[2] * f_out
W, Hh = int(x1 - x0), int(y1 - y0)
T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]])
Hrect = T @ np.array([[f_out, 0, 0], [0, f_out, 0], [0, 0, 1]]) @ R @ np.linalg.inv(K)
print("rectified canvas", W, Hh, "px =", round(W * MM_PX), "x", round(Hh * MM_PX), "mm; y=0 is the wall junction")
def warp(i):
    Hi = np.eye(3) if i == REF else np.linalg.inv(homog(REF, i)[0])
    img = lin[i]; lum = img.mean(2); s = (img.max(2) - img.min(2)) / (img.max(2) + 1e-6)
    valid = ((s > 0.3) & (lum > 0.06)).astype(np.uint8)        # wood only: drops wall, gap shadow, black board
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))   # keep specks/scratches inside the wood
    valid = cv2.erode(valid, np.ones((31, 31), np.uint8)); valid[:8] = 0; valid[-8:] = 0; valid[:, :8] = 0; valid[:, -8:] = 0
    M = Hrect @ Hi
    wi = cv2.warpPerspective(img.astype(np.float32), M, (W, Hh), flags=cv2.INTER_LANCZOS4)
    wv = cv2.warpPerspective(valid.astype(np.float32), M, (W, Hh), flags=cv2.INTER_LINEAR)
    return wi, cv2.erode((wv > 0.999).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0, cv2.Laplacian(g8(img), cv2.CV_64F).var()
ref_i, ref_v, _ = warp(REF)
acc = np.zeros((Hh, W, 3)); wsum = np.zeros((Hh, W))
for i in SET:
    wi, wv, sharp = warp(i)
    both = wv & ref_v
    gain = np.median(ref_i[both] / np.maximum(wi[both], 1e-4), axis=0)   # exposure/WB drift between frames
    # feathered weight so frame footprints do not leave seams
    w = cv2.distanceTransform(wv.astype(np.uint8), cv2.DIST_L2, 3); w = np.clip(w / 80.0, 0, 1) * sharp
    acc += wi * gain * w[..., None]; wsum += w
    print(f"frame {i}: sharp {sharp:.0f} gain {gain.round(3)}")
ex_lin = acc / np.maximum(wsum, 1e-9)[..., None]; mask = wsum > 0.05 * wsum.max()
np.save("exemplar_lin.npy", ex_lin.astype(np.float32)); np.save("exemplar_mask.npy", mask)
np.save("rect_meta.npy", dict(n=n, R=R, Hrect=Hrect, f_out=f_out, junction=(a, b)), allow_pickle=True)
prev = (srgb(ex_lin / np.percentile(ex_lin[mask], 99.5) * 0.9) * 255).astype(np.uint8); prev[~mask] = (40, 0, 60)
cv2.imwrite("exemplar_preview.jpg", cv2.resize(prev[..., ::-1], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
