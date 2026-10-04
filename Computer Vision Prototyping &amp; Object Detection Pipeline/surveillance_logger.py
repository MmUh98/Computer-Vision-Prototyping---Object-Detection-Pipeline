"""
Lab 03 - Task 8: Edge-Triggered Webcam Surveillance System with Log Generation

When a high-priority object (default: person, cell phone) is detected with
confidence > 0.65 the system:
    * saves the frame to alerts/YYYYMMDD_HHMMSS.jpg
    * appends one row per qualifying detection to events.log (CSV)
    * flashes a red border + "EVENT LOGGED" banner on the live stream

events.log columns:  Timestamp, ClassName, Confidence, X1, Y1, X2, Y2, Snapshot

Usage:
    python surveillance_logger.py                                   # webcam 0, q to quit
    python surveillance_logger.py --classes person "cell phone"     # choose priority classes
    python surveillance_logger.py --source ../media/vtest.avi --no-display --max-frames 250 --cooldown 2
"""
import argparse
import csv
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

CONF_THRESHOLD = 0.65
ALERT_DIR = "alerts"
LOG_FILE = "events.log"
HEADER = ["Timestamp", "ClassName", "Confidence", "X1", "Y1", "X2", "Y2", "Snapshot"]
INDICATOR_SECONDS = 1.0
WINDOW = "Surveillance  [q=quit]"


class EventLogger:
    """Writes snapshots + CSV log rows; enforces a cooldown between events."""

    def __init__(self, alert_dir=ALERT_DIR, log_file=LOG_FILE, cooldown=1.0):
        self.alert_dir, self.log_file, self.cooldown = alert_dir, log_file, cooldown
        self._last_event = 0.0
        os.makedirs(alert_dir, exist_ok=True)
        if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
            with open(log_file, "w", newline="") as f:
                csv.writer(f).writerow(HEADER)

    def ready(self, now):
        return now - self._last_event >= self.cooldown

    def _unique_path(self, stamp):
        path = os.path.join(self.alert_dir, f"{stamp}.jpg")
        n = 1
        while os.path.exists(path):                       # guard against same-second collisions
            path = os.path.join(self.alert_dir, f"{stamp}_{n}.jpg")
            n += 1
        return path

    def log(self, frame, detections, now):
        """Save snapshot and append rows. Returns snapshot path."""
        ts = datetime.now()
        stamp = ts.strftime("%Y%m%d_%H%M%S")
        path = self._unique_path(stamp)
        cv2.imwrite(path, frame)
        with open(self.log_file, "a", newline="") as f:
            w = csv.writer(f)
            for d in detections:
                x1, y1, x2, y2 = d["box"]
                w.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), d["name"],
                            f'{d["conf"]:.3f}', x1, y1, x2, y2, os.path.basename(path)])
        self._last_event = now
        return path


def high_priority(result, names, target_names, thr=CONF_THRESHOLD):
    dets = []
    for b in result.boxes:
        name, conf = names[int(b.cls[0])], float(b.conf[0])
        if name in target_names and conf > thr:           # strictly greater than 0.65
            dets.append({"name": name, "conf": conf,
                         "box": tuple(int(v) for v in b.xyxy[0].tolist())})
    return dets


def draw_boxes(frame, dets):
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(out, f'{d["name"]} {d["conf"]:.2f}', (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_indicator(frame, count):
    """Visual cue shown while an event has just been logged."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
    cv2.rectangle(frame, (w - 260, 10), (w - 10, 52), (0, 0, 255), -1)
    cv2.putText(frame, f"EVENT LOGGED #{count}", (w - 250, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def run(args):
    model = YOLO(args.model)
    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.source!r}")
    logger = EventLogger(cooldown=args.cooldown)
    targets, events, frames, indicator_until = set(args.classes), 0, 0, 0.0
    print(f"Watching for {sorted(targets)} with confidence > {args.conf}. Log: {LOG_FILE}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            result = model.predict(frame, conf=0.25, device=args.device, verbose=False)[0]
            dets = high_priority(result, model.names, targets, args.conf)
            vis = draw_boxes(frame, dets)

            now = time.time()
            if dets and logger.ready(now):
                path = logger.log(vis, dets, now)          # snapshot has boxes, no indicator
                events += 1
                indicator_until = now + INDICATOR_SECONDS
                print(f"[EVENT {events}] {os.path.basename(path)}  "
                      + ", ".join(f'{d["name"]}:{d["conf"]:.2f}' for d in dets))
            if now < indicator_until:
                draw_indicator(vis, events)

            if not args.no_display:
                cv2.imshow(WINDOW, vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Done. {frames} frames processed, {events} events logged.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--classes", nargs="+", default=["person", "cell phone"])
    ap.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--cooldown", type=float, default=1.0, help="min seconds between logged events")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
