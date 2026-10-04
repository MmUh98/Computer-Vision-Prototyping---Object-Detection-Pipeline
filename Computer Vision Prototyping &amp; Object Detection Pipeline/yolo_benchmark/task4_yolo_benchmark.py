"""
Lab 03 - Task 4: Multi-Model YOLO Comparative Benchmarking

Loads yolov8n (nano) and yolov8m (medium), runs both on the same image, extracts
boxes / confidences / class labels, measures inference latency, draws detections
with cv2.rectangle + cv2.putText, and prints a comparison table.

Usage:
    python task4_yolo_benchmark.py --image test_image.jpg --runs 10
    python task4_yolo_benchmark.py --models yolov8n.pt yolov8m.pt --conf 0.25 --device cpu
"""
import argparse
import time
from collections import Counter

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO


def color_for(cls_id):
    rng = np.random.default_rng(cls_id * 7 + 3)
    return tuple(int(v) for v in rng.integers(60, 255, 3))


def extract_detections(result, names):
    """Return list of dicts: class_id, label, confidence, x1,y1,x2,y2."""
    out = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cid = int(box.cls[0])
        out.append({"class_id": cid, "label": names[cid], "confidence": float(box.conf[0]),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out


def draw_detections(img_bgr, dets, title):
    vis = img_bgr.copy()
    for d in dets:
        p1, p2 = (int(d["x1"]), int(d["y1"])), (int(d["x2"]), int(d["y2"]))
        col = color_for(d["class_id"])
        cv2.rectangle(vis, p1, p2, col, 2)
        text = f'{d["label"]} {d["confidence"]:.2f}'
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y0 = max(p1[1], th + 6)
        cv2.rectangle(vis, (p1[0], y0 - th - 6), (p1[0] + tw + 4, y0 + base - 4), col, -1)
        cv2.putText(vis, text, (p1[0] + 2, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def benchmark(model_path, img, conf, device, runs, warmup=2, imgsz=640):
    model = YOLO(model_path)
    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    for _ in range(warmup):                                       # warm-up (not timed)
        model.predict(img, conf=conf, device=device, imgsz=imgsz, verbose=False)
    times, result = [], None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = model.predict(img, conf=conf, device=device, imgsz=imgsz, verbose=False)[0]
        times.append((time.perf_counter() - t0) * 1000)           # wall-clock ms, end-to-end
    dets = extract_detections(result, model.names)
    return {"model": model_path, "params_m": params_m, "times": times,
            "speed": result.speed, "dets": dets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="test_image.jpg")
    ap.add_argument("--models", nargs=2, default=["yolov8n.pt", "yolov8m.pt"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    print(f"Image {args.image} {img.shape}; conf>={args.conf}; device={args.device}; runs={args.runs}\n")

    results, rows, vis_imgs = [], [], []
    for mp in args.models:
        r = benchmark(mp, img, args.conf, args.device, args.runs)
        results.append(r)
        dets = r["dets"]
        counts = Counter(d["label"] for d in dets)
        t = np.array(r["times"])
        rows.append({
            "Model": mp.replace(".pt", ""),
            "Params (M)": round(r["params_m"], 1),
            "Latency mean (ms)": round(t.mean(), 1),
            "Latency std (ms)": round(t.std(), 1),
            "Latency min (ms)": round(t.min(), 1),
            "FPS (1/mean)": round(1000 / t.mean(), 2),
            "Detections": len(dets),
            "Classes found": len(counts),
            "Class counts": ", ".join(f"{k}:{v}" for k, v in counts.most_common()),
            "Avg confidence": round(float(np.mean([d["confidence"] for d in dets])), 3) if dets else 0.0,
        })
        name = mp.replace(".pt", "")
        vis = draw_detections(img, dets, f"{name}  {t.mean():.0f} ms  {len(dets)} objs")
        cv2.imwrite(f"detections_{name}.jpg", vis)
        vis_imgs.append((name, vis))

        print(f"--- {name}: detections (class, conf, x1,y1,x2,y2) ---")
        for d in dets:
            print(f'  {d["label"]:<12} {d["confidence"]:.3f}  '
                  f'({d["x1"]:.0f},{d["y1"]:.0f},{d["x2"]:.0f},{d["y2"]:.0f})')
        print()

    df = pd.DataFrame(rows)
    df.to_csv("comparison_table.csv", index=False)
    with open("comparison_table.md", "w") as f:
        f.write(df.to_markdown(index=False) if _has_tabulate() else df.to_string(index=False))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 60)
    print("=== Comparison table ===")
    print(df.to_string(index=False))

    # dual visual output
    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    for a, (name, vis), row in zip(ax, vis_imgs, rows):
        a.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        a.set_title(f'{name}: {row["Detections"]} detections, {row["Latency mean (ms)"]} ms, '
                    f'avg conf {row["Avg confidence"]}', fontweight="bold")
        a.set_xlabel("x (px)"); a.set_ylabel("y (px)")
    fig.suptitle("YOLOv8 Nano vs Medium - same image, same threshold", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("task4_side_by_side.png", dpi=130)
    print("\nSaved: detections_*.jpg, task4_side_by_side.png, comparison_table.csv/.md")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
