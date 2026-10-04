"""
Lab 03 - Task 2: Industrial Defect Detection via Color-Space Transformation & Thresholding

Pipeline: BGR -> HSV & LAB (cv2.cvtColor) -> cv2.inRange on HSV bounds
          -> refine mask (cv2.threshold + morphology) -> cv2.bitwise_and with original.

Usage:
    python task2_defect_detection.py --image metal_part.jpg
    python task2_defect_detection.py          # synthetic rusted-metal demo image

Tuning the HSV bounds (OpenCV ranges: H 0-179, S 0-255, V 0-255):
    rust / orange-brown  : H 5-25,  S 100-255, V 50-255   (default below)
    red defect           : H 0-10 and 170-179 (two ranges)
    green target         : H 35-85
"""
import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Default HSV bounds for rust (change for other defects / colours)
HSV_LOWER = np.array([5, 100, 50], dtype=np.uint8)
HSV_UPPER = np.array([25, 255, 255], dtype=np.uint8)


# --------------------------------------------------------------------------- #
def make_synthetic_rusted_part(path, size=(800, 600), seed=3):
    """Brushed-metal plate with bolts and rust patches (demo only)."""
    w, h = size
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), (165, 160, 155), np.uint8)               # grey metal
    base = np.clip(base + rng.normal(0, 7, base.shape), 0, 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (1, 15), 0)                           # brushed look
    cv2.rectangle(base, (40, 40), (w - 40, h - 40), (120, 118, 115), 6)  # edge frame
    for p in [(90, 90), (w - 90, 90), (90, h - 90), (w - 90, h - 90)]:   # bolts
        cv2.circle(base, p, 22, (110, 108, 105), -1)
        cv2.circle(base, p, 22, (70, 70, 70), 2)
    rust_layer = np.zeros_like(base)
    for _ in range(5):                                                   # rust blotches
        c = (int(rng.integers(150, w - 150)), int(rng.integers(120, h - 120)))
        axes = (int(rng.integers(30, 80)), int(rng.integers(20, 55)))
        cv2.ellipse(rust_layer, c, axes, float(rng.integers(0, 180)), 0, 360,
                    (25, 85, 165), -1)                                   # BGR orange-brown
    rust_layer = cv2.GaussianBlur(rust_layer, (25, 25), 0)
    mask = (rust_layer.sum(axis=2) > 30)[..., None]
    texture = rng.normal(0, 14, base.shape)
    img = np.where(mask, np.clip(rust_layer.astype(np.float32) * 1.15 + texture, 0, 255), base)
    cv2.imwrite(path, img.astype(np.uint8))
    return path


# --------------------------------------------------------------------------- #
def segment_defects(bgr, lower=HSV_LOWER, upper=HSV_UPPER):
    """Return dict with every intermediate stage."""
    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)              # suppress sensor noise first
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)          # Task step 1
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    raw_mask = cv2.inRange(hsv, lower, upper)                # Task step 3: 0 / 255 mask

    # Refinement: binary threshold removes any intermediate values, then
    # morphology removes specks (open) and closes small holes (close).
    _, binary = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    isolated = cv2.bitwise_and(bgr, bgr, mask=binary)        # Task step 4
    return {"hsv": hsv, "lab": lab, "raw_mask": raw_mask, "mask": binary, "isolated": isolated}


def defect_report(mask):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = areas[areas >= 100]                               # ignore tiny speckles
    pct = 100.0 * mask.astype(bool).sum() / mask.size
    return {"regions": int(len(keep)), "defect_pixels": int(mask.astype(bool).sum()),
            "defect_area_pct": round(float(pct), 2)}


# --------------------------------------------------------------------------- #
def plot_grid(bgr, res, out_path):
    rgb = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    H, S, V = cv2.split(res["hsv"])
    L, A, B = cv2.split(res["lab"])

    fig, ax = plt.subplots(3, 4, figsize=(17, 11))
    items = [
        (rgb(bgr), "Original (BGR -> shown as RGB)", None),
        (H, "HSV: Hue channel", "hsv"),
        (S, "HSV: Saturation channel", "gray"),
        (V, "HSV: Value channel", "gray"),
        (L, "LAB: L (lightness)", "gray"),
        (A, "LAB: a (green-red)", "RdYlGn_r"),
        (B, "LAB: b (blue-yellow)", "YlGnBu_r"),
        (res["raw_mask"], "cv2.inRange raw mask", "gray"),
        (res["mask"], "Refined binary mask\n(threshold + morphology)", "gray"),
        (rgb(res["isolated"]), "Isolated defects\n(bitwise_and)", None),
    ]
    for a, (im, title, cmap) in zip(ax.ravel(), items):
        a.imshow(im, cmap=cmap)
        a.set_title(title, fontsize=10, fontweight="bold")
        a.set_xlabel("x (px)")
        a.set_ylabel("y (px)")

    # overlay of detected contours on original for clarity
    overlay = bgr.copy()
    cnts, _ = cv2.findContours(res["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, [c for c in cnts if cv2.contourArea(c) >= 100], -1, (0, 255, 0), 2)
    ax.ravel()[10].imshow(rgb(overlay))
    ax.ravel()[10].set_title("Defect contours on original", fontsize=10, fontweight="bold")
    ax.ravel()[10].set_xlabel("x (px)"); ax.ravel()[10].set_ylabel("y (px)")

    # HSV bounds text panel
    t = ax.ravel()[11]
    t.axis("off")
    rep = defect_report(res["mask"])
    t.text(0.0, 0.9,
           f"HSV lower: {HSV_LOWER.tolist()}\nHSV upper: {HSV_UPPER.tolist()}\n\n"
           f"Defect regions : {rep['regions']}\n"
           f"Defect pixels  : {rep['defect_pixels']}\n"
           f"Defect area    : {rep['defect_area_pct']} %",
           va="top", family="monospace", fontsize=11)
    t.set_title("Detection summary", fontsize=10, fontweight="bold")

    fig.suptitle("Industrial Defect Isolation: BGR -> HSV/LAB -> inRange -> Threshold -> bitwise_and",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    print("Figure saved to:", out_path)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default="task2_output.png")
    args = ap.parse_args()

    path = args.image or make_synthetic_rusted_part("synthetic_rusted_part.jpg")
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    print("Image:", path, bgr.shape)

    res = segment_defects(bgr)
    print("Report:", defect_report(res["mask"]))
    cv2.imwrite("task2_isolated.png", res["isolated"])
    plot_grid(bgr, res, args.out)


if __name__ == "__main__":
    main()
