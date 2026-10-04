"""
Lab 03 - Task 1: Automated Preprocessing Pipeline for Traffic Camera Surveillance

Pipeline:  load (cv2.imread) -> letterbox resize to 640x640 -> normalise to [0, 1]
           -> side-by-side matplotlib display + tensor statistics.

Usage:
    python task1_preprocessing.py --image path/to/traffic.jpg
    python task1_preprocessing.py            # no image given: uses a synthetic
                                             # 1920x1080 traffic scene for demo
"""
import argparse
import os

import cv2
import matplotlib

matplotlib.use("Agg")  # works headless; remove this line to open a window
import matplotlib.pyplot as plt
import numpy as np

TARGET_SIZE = 640          # model input: 640 x 640
PAD_COLOR = (114, 114, 114)  # standard YOLO grey padding


# --------------------------------------------------------------------------- #
# Step 0 (demo only): synthetic traffic scene so the script runs without data
# --------------------------------------------------------------------------- #
def make_synthetic_traffic_scene(path, width=1920, height=1080, seed=7):
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = (200, 170, 130)                                   # sky (BGR)
    cv2.rectangle(img, (0, int(height * 0.35)), (width, height), (70, 70, 70), -1)  # road
    for x in range(0, width, 160):                             # lane markings
        cv2.rectangle(img, (x, int(height * 0.68)), (x + 80, int(height * 0.70)),
                      (230, 230, 230), -1)
    palette = [(30, 30, 200), (200, 60, 30), (40, 160, 40), (220, 220, 220), (30, 200, 220)]
    for _ in range(9):                                         # cars
        w, h = int(rng.integers(150, 260)), int(rng.integers(70, 110))
        x = int(rng.integers(0, width - w))
        y = int(rng.choice([height * 0.5, height * 0.62, height * 0.78]))
        cv2.rectangle(img, (x, y), (x + w, y + h), palette[int(rng.integers(len(palette)))], -1)
        cv2.rectangle(img, (x + w // 5, y - h // 2), (x + 4 * w // 5, y), (50, 50, 50), -1)
        for cx in (x + w // 4, x + 3 * w // 4):
            cv2.circle(img, (cx, y + h), h // 4, (15, 15, 15), -1)
    # simulate uneven exposure + sensor noise (the problem the pipeline addresses)
    gradient = np.linspace(0.6, 1.2, width, dtype=np.float32)[None, :, None]
    img = np.clip(img.astype(np.float32) * gradient, 0, 255)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)
    return path


# --------------------------------------------------------------------------- #
# Step 1: load
# --------------------------------------------------------------------------- #
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR, uint8
    if img is None:
        raise FileNotFoundError(f"cv2.imread could not read: {path}")
    return img


# --------------------------------------------------------------------------- #
# Step 2: aspect-ratio-preserving resize (letterbox)
# --------------------------------------------------------------------------- #
def letterbox(img, size=TARGET_SIZE, color=PAD_COLOR):
    """Resize so the longer side == size, then pad to a size x size square.

    Returns (image, scale, (pad_x, pad_y)); scale/pad let you map detections
    made on the 640x640 image back onto the original frame.
    """
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    # INTER_AREA is best for shrinking; INTER_LINEAR/CUBIC for enlarging
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    pad_x, pad_y = size - new_w, size - new_h
    top, bottom = pad_y // 2, pad_y - pad_y // 2
    left, right = pad_x // 2, pad_x - pad_x // 2
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return out, scale, (left, top)


# --------------------------------------------------------------------------- #
# Step 3: normalise [0,255] -> [0.0,1.0]
# --------------------------------------------------------------------------- #
def normalize(img_uint8):
    return img_uint8.astype(np.float32) / 255.0


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def tensor_stats(name, arr):
    return (f"{name:<28} shape={str(arr.shape):<16} dtype={str(arr.dtype):<8} "
            f"min={arr.min():<8.4f} max={arr.max():<8.4f} "
            f"mean={arr.mean():<8.4f} std={arr.std():.4f}")


# --------------------------------------------------------------------------- #
# Step 4: visualisation
# --------------------------------------------------------------------------- #
def visualize(original_bgr, letterboxed_bgr, normalized_bgr, out_path):
    to_rgb = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2RGB)  # matplotlib expects RGB
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    panels = [
        (axes[0, 0], to_rgb(original_bgr),
         f"Original ({original_bgr.shape[1]}x{original_bgr.shape[0]}, uint8)"),
        (axes[0, 1], to_rgb(letterboxed_bgr),
         f"Letterboxed {TARGET_SIZE}x{TARGET_SIZE} (uint8)"),
        (axes[1, 0], to_rgb(normalized_bgr),
         f"Normalized {TARGET_SIZE}x{TARGET_SIZE} (float32, [0,1])"),
    ]
    for ax, im, title in panels:
        ax.imshow(im)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Width (pixels)")
        ax.set_ylabel("Height (pixels)")

    # histogram: shows value range moving from 0-255 to 0-1
    ax = axes[1, 1]
    gray_o = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY).ravel() / 255.0
    gray_n = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2GRAY).ravel()
    ax.hist(gray_o, bins=64, range=(0, 1), alpha=0.55, label="Original (scaled for comparison)", density=True)
    ax.hist(gray_n, bins=64, range=(0, 1), alpha=0.55, label="Preprocessed", density=True)
    ax.set_title("Intensity Distribution (grayscale)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Normalized intensity")
    ax.set_ylabel("Density")
    ax.legend()

    fig.suptitle("Traffic Camera Preprocessing Pipeline: Load -> Letterbox -> Normalize",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved to: {out_path}")
    return fig


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="path to traffic scene image")
    ap.add_argument("--out", default="task1_output.png")
    args = ap.parse_args()

    image_path = args.image
    if image_path is None:
        image_path = make_synthetic_traffic_scene("synthetic_traffic.jpg")
        print("No --image given; generated synthetic scene:", image_path)

    original = load_image(image_path)
    boxed, scale, (pad_x, pad_y) = letterbox(original)
    normed = normalize(boxed)

    print("\n=== Pixel Tensor Statistics ===")
    print(tensor_stats("Original", original))
    print(tensor_stats("Letterboxed (uint8)", boxed))
    print(tensor_stats("Normalized (float32)", normed))
    print(f"\nLetterbox scale = {scale:.4f}, padding (left, top) = ({pad_x}, {pad_y})")
    print(f"Check: range within [0.0, 1.0]? {normed.min() >= 0.0 and normed.max() <= 1.0}")

    visualize(original, boxed, normed, args.out)
    if os.environ.get("DISPLAY"):
        matplotlib.use("TkAgg", force=True)
        plt.show()


if __name__ == "__main__":
    main()
