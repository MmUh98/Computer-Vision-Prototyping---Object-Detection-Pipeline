# Lab 03 – Computer Vision Prototyping & Object Detection Pipeline

Classical image processing with OpenCV and deep-learning object detection with YOLOv8, applied to four real-world scenarios: traffic monitoring, industrial quality control, medical imaging and autonomous-navigation benchmarking.

**Author:** Muhammad Mansoor Ul Haq

---

## Contents

| Task | Application | Techniques | Main file | Deliverable |
|---|---|---|---|---|
| 1 | Traffic camera preprocessing | `imread`, letterbox `resize`, [0,1] normalisation, Matplotlib | `task1_preprocessing.py` | Multi-stage figure + tensor statistics |
| 2 | Industrial defect detection | BGR→HSV/LAB, `inRange`, `threshold`, morphology, `bitwise_and` | `task2_defect_detection.py` | Grid of channels, mask and isolated defects |
| 3 | Medical image edge detection | Gaussian vs Median filtering, Sobel, Canny, PSNR/SSIM/F1 | `task3_medical_edge_detection.ipynb` | Quantitative filter comparison notebook |
| 4 | YOLO benchmarking | YOLOv8n vs YOLOv8m, latency, `rectangle` / `putText` | `task4/task4_yolo_benchmark.py` | Side-by-side detections + comparison table |

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

## Notes

* All demo outputs were produced on CPU. Task 1–3 demo images are synthetic; replace them with real images for submission.
* Matplotlib uses the headless `Agg` backend so scripts run on servers; figures are saved to PNG files.
* Tasks 5–8 (class-filtered counting, ROI intrusion alerts, live webcam detection, event logging) are not covered in this README.
