import numpy as np, cv2
M2020_709 = np.array([[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]])
def hlg_to_linear709(path):
    """16-bit HLG-encoded BT.2020 RGB png -> scene-linear BT.709 float RGB (HLG scene light, 0..1)."""
    e = cv2.imread(path, cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float64) / 65535.0
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    lin = np.where(e <= 0.5, e * e / 3.0, (np.exp((e - c) / a) + b) / 12.0)
    return np.clip(lin @ M2020_709.T, 0, None)
def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
