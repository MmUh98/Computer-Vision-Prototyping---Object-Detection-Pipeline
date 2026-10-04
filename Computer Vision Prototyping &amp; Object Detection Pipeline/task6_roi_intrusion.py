"""
Lab 03 - Task 6: Smart Security Intrusion Alerting with ROI Bounding Box Analytics

1. Define a polygon ROI with OpenCV.
2. Detect people with YOLO and compute each bounding-box centroid (xc, yc).
3. Test each centroid against the polygon with cv2.pointPolygonTest.
4. ROI drawn RED + "INTRUSION DETECTED" if any person is inside, GREEN otherwise.

Usage:
    python task6_roi_intrusion.py --image surveillance_frame.jpg --out result_intrusion.jpg
    python task6_roi_intrusion.py --image surveillance_frame.jpg \
        --roi "20,420;300,420;300,560;20,560" --out result_clear.jpg
ROI format: "x1,y1;x2,y2;x3,y3;..." in image pixel coordinates (at least 3 points).
"""
import argparse

import cv2
import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0
GREEN, RED, BLUE, WHITE = (0, 200, 0), (0, 0, 255), (255, 120, 0), (255, 255, 255)
DEFAULT_ROI = "350,220;680,220;680,420;350,420"      # demo polygon for the sample frame


def parse_roi(text):
    pts = [tuple(int(float(v)) for v in p.split(",")) for p in text.split(";") if p.strip()]
    if len(pts) < 3:
        raise ValueError("ROI needs at least 3 points")
    return np.array(pts, dtype=np.int32)


def detect_people(model, img, conf=0.4, device="cpu"):
    res = model.predict(img, conf=conf, classes=[PERSON_CLASS_ID], device=device, verbose=False)[0]
    people = []
    for b in res.boxes:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        people.append({"box": (x1, y1, x2, y2), "conf": float(b.conf[0]),
                       "centroid": ((x1 + x2) // 2, (y1 + y2) // 2)})   # (xc, yc)
    return people


def inside_roi(roi_poly, point):
    # measureDist=False -> +1 inside, 0 on edge, -1 outside
    return cv2.pointPolygonTest(roi_poly.reshape(-1, 1, 2), (float(point[0]), float(point[1])), False) >= 0


def draw_result(img, roi_poly, people):
    out = img.copy()
    flags = [inside_roi(roi_poly, p["centroid"]) for p in people]
    intrusion = any(flags)
    roi_color = RED if intrusion else GREEN

    # translucent ROI fill + perimeter
    overlay = out.copy()
    cv2.fillPoly(overlay, [roi_poly], roi_color)
    out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
    cv2.polylines(out, [roi_poly], True, roi_color, 3)

    for p, inside in zip(people, flags):
        x1, y1, x2, y2 = p["box"]
        col = RED if inside else BLUE
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.circle(out, p["centroid"], 4, col, -1)
        cv2.putText(out, f'person {p["conf"]:.2f}', (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    n_in = sum(flags)
    if intrusion:
        text, tcol = f"INTRUSION DETECTED ({n_in})", RED
    else:
        text, tcol = "ZONE CLEAR", GREEN
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    cv2.rectangle(out, (0, 0), (tw + 24, th + 22), (20, 20, 20), -1)
    cv2.putText(out, text, (12, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, tcol, 3, cv2.LINE_AA)
    return out, intrusion, n_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="surveillance_frame.jpg")
    ap.add_argument("--roi", default=DEFAULT_ROI)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="task6_result.jpg")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    roi = parse_roi(args.roi)
    model = YOLO(args.model)

    people = detect_people(model, img, args.conf, args.device)
    result, intrusion, n_in = draw_result(img, roi, people)

    print(f"People detected: {len(people)}; inside ROI: {n_in}")
    for i, p in enumerate(people, 1):
        print(f'  #{i} centroid={p["centroid"]} conf={p["conf"]:.2f} '
              f'inside={inside_roi(roi, p["centroid"])}')
    print("STATUS:", "INTRUSION DETECTED" if intrusion else "ZONE CLEAR")
    cv2.imwrite(args.out, result)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
