# Lab 03 – Computer Vision Prototyping & Object Detection Pipeline

Classical image processing with OpenCV and deep-learning object detection with YOLOv8, applied to eight real-world scenarios: traffic monitoring, industrial quality control, medical imaging, model benchmarking, retail inventory, perimeter security, live video analytics and event logging.

**Author:** Muhammad Mansoor Ul Haq

---

## Contents

| Task | Application | Techniques | Main file | Deliverable |
|---|---|---|---|---|
| 1 | Traffic camera preprocessing | `imread`, letterbox `resize`, [0,1] normalisation, Matplotlib | `task1_preprocessing.py` | Multi-stage figure + tensor statistics |
| 2 | Industrial defect detection | BGR→HSV/LAB, `inRange`, `threshold`, morphology, `bitwise_and` | `task2_defect_detection.py` | Grid of channels, mask and isolated defects |
| 3 | Medical image edge detection | Gaussian vs Median filtering, Sobel, Canny, PSNR/SSIM/F1 | `task3_medical_edge_detection.ipynb` | Quantitative filter comparison notebook |
| 4 | YOLO benchmarking | YOLOv8n vs YOLOv8m, latency, `rectangle` / `putText` | `task4/task4_yolo_benchmark.py` | Side-by-side detections + comparison table |
| 5 | Retail inventory counting | Class-ID filtering, confidence threshold, on-image banner | `task5/task5_retail_counting.py` | Filtered image with item count |
| 6 | Perimeter intrusion alerts | Polygon ROI, centroids, `pointPolygonTest` | `task6/task6_roi_intrusion.py` | Result images with red/green ROI and alert text |
| 7 | Live detection dashboard | `VideoCapture`, FPS meter, key events, resource release | `task7/live_detection.py` | Modular real-time script |
| 8 | Event-triggered surveillance | Confidence trigger, snapshots, CSV log, visual indicator | `task8/surveillance_logger.py` | Script + `alerts/` snapshots + `events.log` |

## Project Structure

```text
lab03/
├── README.md
├── task1_preprocessing.py          # Task 1 script
├── task1_output.png                # Task 1 figure (generated)
├── task1_stats.txt                 # Task 1 tensor statistics (generated)
├── task2_defect_detection.py       # Task 2 script
├── task2_output.png                # Task 2 figure grid (generated)
├── task2_isolated.png              # Task 2 isolated defects (generated)
├── task3_medical_edge_detection.ipynb   # Task 3 notebook (executed)
├── task3_results.csv               # Task 3 results table (generated)
└── task4/
    ├── task4_yolo_benchmark.py     # Task 4 script
    ├── test_image.jpg              # sample image (ultralytics bus.jpg)
    ├── detections_yolov8n.jpg      # annotated output, nano
    ├── detections_yolov8m.jpg      # annotated output, medium
    ├── task4_side_by_side.png      # dual visual output
    └── comparison_table.csv / .md  # benchmark table
├── media/
│   └── vtest.avi                   # pedestrian test video (OpenCV sample) for Tasks 6-8
├── task5/
│   ├── task5_retail_counting.py
│   ├── shelf_sample.jpg            # sample input
│   └── task5_result.jpg            # annotated output
├── task6/
│   ├── task6_roi_intrusion.py
│   ├── surveillance_frame.jpg      # sample input (frame from vtest.avi)
│   ├── result_intrusion.jpg        # ROI red, alert shown
│   └── result_clear.jpg            # ROI green, zone clear
├── task7/
│   ├── live_detection.py
│   └── captures/sample_frame.jpg   # saved frame with FPS overlay
└── task8/
    ├── surveillance_logger.py
    ├── alerts/                     # 14 timestamped snapshots (YYYYMMDD_HHMMSS.jpg)
    ├── events.log                  # CSV log generated during the run
    └── indicator_demo.jpg          # example of the on-stream "EVENT LOGGED" indicator
```

Scripts for Tasks 1 and 2 also create demo inputs (`synthetic_traffic.jpg`, `synthetic_rusted_part.jpg`) when no `--image` is given.

## Setup

Python 3.9+ is recommended.

```bash
pip install opencv-python matplotlib numpy pandas scikit-image jupyter ultralytics tabulate
```

* `scikit-image` is optional (SSIM in Task 3; the notebook runs without it, SSIM is then skipped).
* `ultralytics` pulls in PyTorch. Task 4 runs on CPU by default; pass `--device cuda:0` for a GPU.
* YOLO weights (`yolov8n.pt`, `yolov8m.pt`) download automatically on first run.

## Usage

### Task 1 – Automated Preprocessing Pipeline

```bash
python task1_preprocessing.py --image your_traffic_scene.jpg
```

1. Loads the image with `cv2.imread`.
2. Letterboxes to 640×640 (aspect ratio preserved, grey padding, scale and padding returned so detections can be mapped back to the original frame).
3. Normalises pixel values from [0, 255] to float32 [0.0, 1.0].
4. Plots original, letterboxed, normalised image and an intensity histogram, then prints shape / dtype / min / max / mean / std for each stage.

**Demo result (1920×1080 synthetic scene):** scale 0.3333, padding 140 px top and bottom, normalised range 0.0157 – 1.0.

### Task 2 – Industrial Defect Detection

```bash
python task2_defect_detection.py --image your_part.jpg
```

1. Converts BGR to HSV and LAB (`cv2.cvtColor`).
2. Isolates the target colour using `cv2.inRange`. Default bounds are for rust: `[5,100,50]` to `[25,255,255]` (OpenCV hue range 0–179). **Tune `HSV_LOWER` / `HSV_UPPER` at the top of the script for your own image.**
3. Refines the mask with `cv2.threshold` plus morphological open/close.
4. Applies `cv2.bitwise_and` to isolate the defects, and reports region count and defect area %.

**Demo result:** 3 rust regions, 5.5 % of the image area.

### Task 3 – Edge Detection & Noise Reduction (Medical Imaging)

```bash
jupyter notebook task3_medical_edge_detection.ipynb
```

Set `IMAGE_PATH` in the first code cell to use a real X-ray / CT image; otherwise a synthetic CT-like phantom is used. The clean image is treated as ground truth and two noise types are added: Gaussian (σ = 20) and salt-and-pepper (4 %).

* **Denoising:** Gaussian blur vs Median filter at k = 3, 5, 9, scored with PSNR and SSIM.
* **Sobel:** X and Y gradients plus magnitude, scored with an Edge-to-Background Ratio.
* **Canny:** hysteresis thresholds tuned per filter by grid search to maximise edge F1 (±2 px tolerance) against the ground-truth edge map.
* Results are saved to `task3_results.csv`.

**Demo findings:**

| Observation | Evidence |
|---|---|
| Median filtering handles impulse noise best | Salt & pepper, Median k=3: PSNR 46.6 dB, F1 = 1.00; Gaussian k=5: F1 = 0.79 |
| Filtering is essential before Canny | No filter: F1 = 0.44 (Gaussian noise), 0.28 (salt & pepper) |
| Thresholds must be re-tuned per filter | Gaussian k=3: F1 0.47 at fixed (50, 150) vs 0.97 tuned |

> The synthetic phantom is piecewise-constant, which flatters the median filter. Expect lower absolute scores on real scans.

### Task 4 – YOLO Comparative Benchmarking

```bash
cd task4
python task4_yolo_benchmark.py --image test_image.jpg --runs 10
python task4_yolo_benchmark.py --image your.jpg --models yolov8n.pt yolov8s.pt --conf 0.25 --device cuda:0
```

Loads two YOLO variants, runs inference on the same image (2 warm-up runs, then timed runs), extracts class / confidence / box coordinates, draws results with `cv2.rectangle` and `cv2.putText`, and writes a comparison table.

**Demo result (CPU, conf ≥ 0.25):**

| Model | Params | Latency (mean) | FPS | Detections | Classes | Avg confidence |
|---|---|---|---|---|---|---|
| yolov8n | 3.2 M | 107.3 ms | 9.3 | 6 | person 4, bus 1, stop sign 1 | 0.656 |
| yolov8m | 25.9 M | 645.5 ms | 1.55 | 5 | person 4, bus 1 | 0.892 |

Nano is about 6× faster, while Medium is markedly more confident. Nano's extra "stop sign" (0.26) is a low-confidence detection that disappears at a higher threshold. Latency figures depend on the machine, so only the ratio is meaningful across systems.

### Task 5 – Retail Inventory Monitoring (Class-Filtered Detection)

```bash
cd task5
python task5_retail_counting.py --image shelf.jpg                  # bottles (39) + apples (47), the lab defaults
python task5_retail_counting.py --image shelf.jpg --classes 39     # bottles only
python task5_retail_counting.py --image shelf_sample.jpg --classes 54 --label donut   # included demo
```

YOLO runs once at a low confidence floor; detections are then filtered **in code** by class ID and kept only if confidence > 0.50. Kept items are boxed and numbered, and a banner shows the count ("Target Item Count: N"). Confident detections of non-target classes are listed in the console as ignored.

**Demo result:** 9 donuts counted (confidence 0.87 – 0.89). No bottle/apple shelf photo was available offline, so the included demo uses a tray of donuts (COCO class 54) as a stand-in; the script works identically for classes 39 and 47.

### Task 6 – Intrusion Alerting with ROI Analytics

```bash
cd task6
python task6_roi_intrusion.py --image surveillance_frame.jpg --out result_intrusion.jpg
python task6_roi_intrusion.py --image surveillance_frame.jpg --roi "20,420;300,420;300,560;20,560" --out result_clear.jpg
```

The ROI is a polygon given as `"x,y;x,y;..."` (pixel coordinates). For each detected person the centroid `(xc, yc)` is the box centre, tested with `cv2.pointPolygonTest`. If any centroid is inside, the ROI is drawn **red** with an "INTRUSION DETECTED (n)" alert; otherwise it is **green** with "ZONE CLEAR". People inside the ROI get red boxes, others blue.

**Demo result:** 7 people detected; with the central ROI, 3 centroids fall inside, giving an intrusion alert. With an empty corner ROI, 0 fall inside and the zone is clear.

### Task 7 – Real-Time Webcam Detection Dashboard

```bash
cd task7
python live_detection.py                        # webcam 0
python live_detection.py --source 1             # another camera
python live_detection.py --source video.mp4     # video file instead of a camera
```

Keys (click the window first): **q** quits, **s** saves the current annotated frame to `captures/`. The top-left corner shows a smoothed FPS counter and object count. The code is modular: `FPSMeter`, `open_capture`, `annotate`, `handle_key` and `run`. Resources are always freed in a `finally` block with `cap.release()` and `cv2.destroyAllWindows()`. A clear error is raised if the camera cannot be opened.

**Tested on:** `media/vtest.avi` in headless mode (`--no-display --max-frames 60`), about 9.8 FPS on CPU with YOLOv8n. The key handler was unit-tested separately. The live webcam window itself could not be exercised in the build environment (no camera or display), so run it once on your machine.

### Task 8 – Edge-Triggered Surveillance with Log Generation

```bash
cd task8
python surveillance_logger.py                                       # webcam, person + cell phone
python surveillance_logger.py --classes "cell phone" --conf 0.65
python surveillance_logger.py --source ../media/vtest.avi --no-display --max-frames 250 --cooldown 2
```

When a target class is detected with confidence **> 0.65**:
* the annotated frame is saved to `alerts/YYYYMMDD_HHMMSS.jpg` (a cooldown, 1 s by default, keeps names unique; a numeric suffix is added on any collision);
* one row per qualifying detection is appended to `events.log` with columns `Timestamp, ClassName, Confidence, X1, Y1, X2, Y2, Snapshot`;
* the live stream shows a red border and an "EVENT LOGGED #n" banner for one second.

**Demo result:** 250 video frames produced 14 snapshots and 75 log rows; the lowest logged confidence is 0.663, confirming the threshold. The test video contains pedestrians but no phones, so only the `person` class fired.

## Notes

* All demo outputs were produced on CPU. Task 1–3 demo images are synthetic and Task 5 uses a donut tray; replace them with real images for submission.
* Tasks 6–8 use `media/vtest.avi` (an OpenCV sample video) as stand-in surveillance footage.
* Matplotlib uses the headless `Agg` backend so scripts run on servers; figures are saved to PNG files.
* Model weights are not stored in the folders; ultralytics downloads `yolov8n.pt` automatically on first run.
