"""
Lab 03 - Task 5: Retail Inventory Monitoring with Class-Filtered Object Detection

Runs YOLO on a shelf image, keeps only the target COCO class IDs, counts items above
a 0.50 confidence threshold, and overlays boxes plus a summary banner.

COCO IDs of interest:  39 = bottle, 47 = apple, 49 = orange, 46 = banana, 54 = donut,
                       41 = cup, 45 = bowl      (full list: model.names)

Usage:
    python task5_retail_counting.py --image shelf.jpg                    # bottles + apples (lab default)
    python task5_retail_counting.py --image shelf.jpg --classes 39       # bottles only
    python task5_retail_counting.py --image shelf_sample.jpg --classes 54 --label donut
"""
import argparse
from collections import Counter

import cv2
from ultralytics import YOLO

DEFAULT_CLASSES = [39, 47]   # bottle, apple
CONF_THRESHOLD = 0.50


def detect_targets(model, img, target_ids, conf=CONF_THRESHOLD, device="cpu"):
    """Run YOLO once at a low threshold, then filter by class ID and confidence in code."""
    res = model.predict(img, conf=0.05, device=device, verbose=False)[0]   # keep everything first
    targets, ignored = [], []
    for b in res.boxes:
        cid, score = int(b.cls[0]), float(b.conf[0])
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        det = {"id": cid, "name": model.names[cid], "conf": score, "box": (x1, y1, x2, y2)}
        if cid in target_ids and score > conf:          # programmatic class + confidence filter
            targets.append(det)
        elif score > conf:
            ignored.append(det)                          # confident detections we deliberately drop
    return targets, ignored


def annotate(img, targets, banner_text):
    out = img.copy()
    for i, d in enumerate(targets, 1):
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, f'{i}:{d["conf"]:.2f}', (x1 + 2, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    # banner across the top
    h, w = out.shape[:2]
    bar_h = max(40, h // 12)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (30, 30, 30), -1)
    out = cv2.addWeighted(overlay, 0.75, out, 0.25, 0)
    scale = bar_h / 50.0
    cv2.putText(out, banner_text, (12, int(bar_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="shelf_sample.jpg")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--classes", type=int, nargs="+", default=DEFAULT_CLASSES)
    ap.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--label", default=None, help="text shown in banner (default: 'Target Item')")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="task5_result.jpg")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    model = YOLO(args.model)
    names = [model.names[c] for c in args.classes]
    print(f"Target classes: {dict(zip(args.classes, names))}; conf > {args.conf}")

    targets, ignored = detect_targets(model, img, set(args.classes), args.conf, args.device)
    per_class = Counter(d["name"] for d in targets)
    print(f"\nTarget Item Count: {len(targets)}   per class: {dict(per_class)}")
    for i, d in enumerate(targets, 1):
        print(f'  #{i:<2} {d["name"]:<8} conf={d["conf"]:.3f} box={d["box"]}')
    print(f"Ignored non-target detections (conf > {args.conf}): "
          f"{dict(Counter(d['name'] for d in ignored)) or 'none'}")

    banner = f"Target Item Count: {len(targets)}"
    if args.label:
        banner = f"{args.label.capitalize()} Count: {len(targets)}"
    result = annotate(img, targets, banner)
    cv2.imwrite(args.out, result)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
