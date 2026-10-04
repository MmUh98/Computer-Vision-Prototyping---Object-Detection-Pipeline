"""
Lab 03 - Task 7: Real-Time Webcam Multi-Object Detection Dashboard

Controls (click the video window first):
    q  - quit
    s  - save current annotated frame to captures/

Usage:
    python live_detection.py                       # webcam 0
    python live_detection.py --source 1            # second camera
    python live_detection.py --source video.mp4    # video file instead of a camera
    python live_detection.py --source ../media/vtest.avi --no-display --max-frames 100   # headless test
"""
import argparse
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

WINDOW = "Live Detection  [q=quit | s=save]"
SAVE_DIR = "captures"


class FPSMeter:
    """Exponentially-smoothed frames-per-second counter."""

    def __init__(self, alpha=0.1):
        self.alpha, self.fps, self._last = alpha, 0.0, None

    def tick(self):
        now = time.perf_counter()
        if self._last is not None:
            inst = 1.0 / max(now - self._last, 1e-9)
            self.fps = inst if self.fps == 0 else self.alpha * inst + (1 - self.alpha) * self.fps
        self._last = now
        return self.fps


def open_capture(source):
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source!r}")
    return cap


def annotate(frame, result, fps, model_name):
    """Draw detections (Ultralytics plot) and an FPS panel in the top-left corner."""
    out = result.plot()                                   # boxes + labels + confidences
    text = f"FPS: {fps:5.1f}"
    cv2.rectangle(out, (0, 0), (170, 62), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(out, f"{model_name}  objs:{len(result.boxes)}", (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_frame(frame, folder=SAVE_DIR):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f")[:-3] + ".jpg")
    cv2.imwrite(path, frame)
    return path


def handle_key(key, frame):
    """Return (should_quit, saved_path_or_None). Pure function -> easy to test."""
    key &= 0xFF
    if key == ord("q"):
        return True, None
    if key == ord("s"):
        return False, save_frame(frame)
    return False, None


def run(args):
    model = YOLO(args.model)
    cap = open_capture(args.source)
    meter = FPSMeter()
    frames, last_vis = 0, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Stream ended / no frame received.")
                break
            result = model.predict(frame, conf=args.conf, device=args.device,
                                   imgsz=args.imgsz, verbose=False)[0]
            fps = meter.tick()
            vis = annotate(frame, result, fps, os.path.basename(args.model))
            last_vis = vis
            frames += 1

            if not args.no_display:
                cv2.imshow(WINDOW, vis)
                quit_now, saved = handle_key(cv2.waitKey(1), vis)
                if saved:
                    print("Saved:", saved)
                if quit_now:
                    break
            if args.max_frames and frames >= args.max_frames:
                break
    finally:                                              # always release resources
        cap.release()
        cv2.destroyAllWindows()
    print(f"Processed {frames} frames; final smoothed FPS = {meter.fps:.1f}")
    if args.no_display and last_vis is not None:
        print("Saved sample frame:", save_frame(last_vis))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index or video file path")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-display", action="store_true", help="headless mode (no window / keys)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run until q)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
