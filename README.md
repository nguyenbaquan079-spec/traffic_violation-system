# Traffic Violation Detection System

[![CI](https://github.com/nguyenbaquan079-spec/traffic_violation-system/actions/workflows/ci.yml/badge.svg)](https://github.com/nguyenbaquan079-spec/traffic_violation-system/actions/workflows/ci.yml)

A desktop application that watches traffic footage and automatically records two kinds of
violations — **running a red light** and **riding a motorcycle without a helmet** — then saves
photo evidence to disk and logs everything to a SQLite database.

Built with **YOLO (Ultralytics) + ByteTrack + OpenCV + PySide6**. It works on a video file,
a webcam, or an IP camera (RTSP).

> 🇻🇳 The desktop GUI is in Vietnamese. See [GUI glossary](#gui-glossary-vietnamese--english)
> below, or read the [Vietnamese documentation](docs/README.vi.md).

---

## Contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Getting the models](#getting-the-models)
- [Quick start (GUI)](#quick-start-gui)
- [Batch mode (no GUI)](#batch-mode-no-gui)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Running the tests](#running-the-tests)
- [Limitations & legal notice](#limitations--legal-notice)

---

## What it does

| | |
|---|---|
| 🚦 **Red-light running** | A vehicle is tracked across frames; if it crosses the stop line while the light is red, it is recorded once. |
| 🪖 **No helmet** | A motorcycle rider detected without a helmet for several consecutive frames is recorded once. |
| 🔢 **Licence plate** *(optional)* | A second YOLO model crops the plate and EasyOCR tries to read the characters. |
| 🖼️ **Evidence** | For every violation three JPEGs are saved: full **scene**, **vehicle** crop, and **plate** crop. |
| 🗃️ **Log** | Every violation is a row in `violations.db` (SQLite), and is reloaded into the table on the next start. |

## How it works

```
video frame
   │
   ├─► YOLO11n (COCO) + ByteTracking ──► vehicles with stable IDs
   │                                     (car / motorcycle / bus / truck)
   │
   ├─► HSV colour check inside the light region ──► red / yellow / green
   │        (majority vote over the last 7 frames)
   │
   ├─► helmet model (every N frames) ──► "no helmet" boxes ──► matched to a
   │        motorcycle whose box, expanded upward, contains the rider's head
   │
   └─► rules ──► violation? ──► save scene/vehicle/plate JPEGs + SQLite row
```

- **Red light:** the light state is classified by counting saturated red/yellow/green pixels
  inside the rectangle you draw around the traffic light, then smoothed by a majority vote over
  7 frames. A vehicle violates when the bottom-centre of its bounding box ("foot point") crosses
  your stop line **in the direction of the arrow drawn on screen**, while the light is red.
- **No helmet:** the helmet detector runs every `helmet_every` frames (default: every 2nd frame)
  to save CPU. A "no helmet" detection is assigned to the nearest motorcycle whose box — expanded
  upward by one box height — contains it. To cut down false alarms, a rider must be confirmed for
  `helmet_confirm_frames` consecutive checks (default: 3) before it is recorded.
- **Once per vehicle:** each `(violation type, track ID)` pair is only ever reported once.

## Project layout

```
.
├── main_gui.py        # PySide6 desktop app (the normal way to run it)
├── run_video.py       # headless: process one video -> annotated MP4 + summary
├── download_models.py # fetch the helmet + plate models, auto-fill config.json
├── debug_helmet.py    # diagnostic: why is "no helmet" never triggering?
├── pipeline.py        # ViolationSystem: detection + tracking + rules per frame
├── rules.py           # geometry (line crossing) and HSV traffic-light classifier
├── plate.py           # plate detection (YOLO) + character reading (EasyOCR)
├── storage.py         # writes evidence JPEGs and the SQLite log
├── worker.py          # QThread that runs the pipeline without freezing the GUI
├── config.py          # Settings dataclass, loaded from / saved to config.json
├── config.json        # your regions, thresholds and model paths
├── yolo11n.pt         # COCO vehicle detector (Ultralytics downloads it if missing)
├── models/            # helmet.pt, plate.pt  (see "Getting the models")
├── tests/             # logic tests using fake models — no GPU or weights needed
├── evidence/          # captured violations, one folder per day (git-ignored)
└── violations.db      # SQLite log (git-ignored)
```

> All commands must be run **from the repository root** — `config.json` is loaded from the
> current working directory.

## Requirements

- Python **3.9+** (CI runs the tests on 3.10 and 3.11)
- Windows, Linux or macOS. A GPU helps a lot but is not required.
- ~1.5 GB of disk for the Python dependencies (PyTorch comes in via `ultralytics`)

## Installation

```bash
git clone https://github.com/nguyenbaquan079-spec/traffic_violation-system.git
cd traffic_violation-system

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` installs `ultralytics`, `PySide6`, `numpy`, `huggingface_hub` and `easyocr`.
OCR is optional — if EasyOCR fails to install, the app still runs and simply saves the plate
crop without reading the text.

### Fixing the Qt plugin error on Linux

If the GUI refuses to start with ``Could not load the Qt platform plugin "xcb"``, OpenCV and
PySide6 are fighting over the Qt libraries. Use the headless OpenCV build:

```bash
pip uninstall -y opencv-python opencv-python-headless
pip install opencv-python-headless
```

## Getting the models

On top of the built-in COCO vehicle detector, the system needs two extra models in `models/`.
Both are already committed here, so you can skip this section — it is what you need when you
want to refresh or replace them:

| File | Purpose | Sample source used by the script |
|---|---|---|
| `models/helmet.pt` | rider with/without helmet | `FatimaNoorAI/helmet-detection` (`safetyHelmet.pt`, YOLOv8) |
| `models/plate.pt` | licence plate detection | `Koushim/yolov8-license-plate-detection` (`best.pt`, YOLOv8n, MIT) |

The quickest way to get both — it downloads them and fills in `config.json` for you:

```bash
python download_models.py                    # downloads into models/ and updates config.json
python download_models.py --test photo.jpg   # also runs both models on one image -> test_output.jpg
```

Different helmet models name their classes differently. The script detects the "no helmet"
class automatically; if it cannot, list the class names and set `no_helmet_names` in
`config.json` yourself:

```bash
python -c "from ultralytics import YOLO; print(YOLO('models/helmet.pt').names)"
```

You can swap in any Ultralytics `.pt` model. For real accuracy on your own footage you will
want to fine-tune on local traffic images. **Only download `.pt` files from sources you trust** —
they are PyTorch pickles and execute code when loaded.

## Quick start (GUI)

```bash
python main_gui.py
```

1. **Pick a source** — *Mở video…* (open a file), *Webcam*, or *Camera IP…* (RTSP/HTTP URL).
2. **Chọn vùng đèn** *(pick light region)* — click two opposite corners of a tight box around
   the traffic-light bulb. The tighter the better.
3. **Đặt vạch dừng** *(set stop line)* — click the two ends of the stop line on the road.
   The **arrow** on the line shows which direction counts as running the light; if it points
   the wrong way, tick **Đảo chiều vạch** *(flip line direction)*.
4. **Bắt đầu** *(start)* — click any row in the table to see its scene, vehicle and plate photos.

Both regions are saved to `config.json`, so you only set them once per camera angle.

## Batch mode (no GUI)

To process a recorded video and get an annotated MP4 plus a summary:

```bash
python run_video.py video.mp4 --out result.mp4
python run_video.py video.mp4 --out result.mp4 --max-frames 600   # only the first 600 frames
```

It reuses the `stop_line` and `light_roi` you configured in the GUI and prints a summary telling
you how many frames were seen in each light colour — the fastest way to check whether your light
region is actually correct.

## Configuration

Everything lives in `config.json` (the GUI writes it for you; you can also edit it by hand):

| Key | Default | Meaning |
|---|---|---|
| `vehicle_model` | `yolo11n.pt` | COCO detector for cars/buses/trucks/motorcycles |
| `helmet_model` | `models/helmet.pt` | helmet detector; if the file is missing the rule is disabled |
| `plate_model` | `models/plate.pt` | plate detector; if missing, only vehicle crops are saved |
| `device` | `""` | `""` = auto, `"cpu"` = force CPU, `"0"` = first GPU |
| `imgsz` | `640` | inference resolution — lower is faster, higher is more accurate |
| `conf_vehicle` / `conf_helmet` / `conf_plate` | `0.4` / `0.4` / `0.3` | confidence thresholds |
| `no_helmet_names` | see file | which class names from your helmet model mean "no helmet" |
| `helmet_confirm_frames` | `3` | consecutive detections required before reporting |
| `helmet_every` | `2` | run the helmet model every N frames |
| `stop_line` | `null` | `[[x1,y1],[x2,y2]]` — set in the GUI |
| `light_roi` | `null` | `[x1,y1,x2,y2]` — set in the GUI |
| `flip_direction` | `false` | reverse which way counts as crossing |
| `use_ocr` | `true` | try to read plate characters with EasyOCR |
| `evidence_dir` / `db_path` | `evidence` / `violations.db` | where output is written |

### GUI glossary (Vietnamese → English)

| Button / column | Meaning |
|---|---|
| Mở video… | Open video file |
| Webcam | Use the built-in camera |
| Camera IP… | Enter an RTSP/HTTP stream URL |
| ▶ Bắt đầu / ■ Dừng | Start / Stop |
| Đặt vạch dừng | Set the stop line (click 2 points) |
| Chọn vùng đèn | Pick the traffic-light region (click 2 corners) |
| Đảo chiều vạch | Flip the line direction |
| Thời gian / Vi phạm / Xe # / Biển số | Time / Violation / Vehicle # / Plate |

## Troubleshooting

**Nothing is detected at all**
: Check that `models/helmet.pt` exists and that you are running from the repository root.
  The status bar prints which models loaded.

**"No helmet" is never reported**
: Run the diagnostic — it draws exactly what the model sees and tells you which step fails:

  ```bash
  python debug_helmet.py video.mp4 --conf 0.1 --max-frames 600 --out debug_helmet.mp4
  ```

  Most often the cause is a class-name mismatch: your model's class must match
  `no_helmet_names` in `config.json` (`normalize_name` lowercases and ignores spaces, `_` and
  `-`, so `Without_Helmet` and `without helmet` are equivalent).

**Red light is never reported (or reports constantly)**
: The arrow on the stop line shows the direction that counts — flip it if needed. Then run
  `run_video.py` and read the "frames per light colour" summary: if it says `unknown` for most
  frames, your `light_roi` box is not tight around the bulb.

**It is too slow**
: Set `device` to `"0"` to use a GPU, lower `imgsz` to `416` or `320`, and raise `helmet_every`
  to `4` or more. Without a GPU, expect a few FPS.

**Plate text is wrong or empty**
: OCR needs a large, sharp plate: mount the camera on the side the plates face, use a high
  resolution, and keep the stop line close to the camera. Treat the text as a hint, not as fact.

## Running the tests

The logic tests swap YOLO for a fake detector, so they need only NumPy and OpenCV — no weights,
no GPU, no Qt:

```bash
pip install numpy opencv-python-headless
python -m tests.test_logic
```

They cover the geometry (line crossing, direction flipping), the HSV traffic-light classifier,
the red-light rule end to end, and the helmet confirmation logic. The same command runs in CI on
every push and pull request.

## Limitations & legal notice

- This is a **baseline / demo system**. It has not been validated against ground-truth data;
  test it on your own footage and tune the thresholds in `config.json` before trusting it.
- Tracking IDs can change when a vehicle is occluded, so a vehicle may occasionally be recorded
  twice — or missed entirely.
- Plate OCR quality is limited, especially at a distance or at an angle.
- **Privacy:** evidence photos can contain readable licence plates and faces of identifiable
  people. For that reason `evidence/` and `violations.db` are listed in `.gitignore` — do not
  commit captures from real cameras, and store them according to your local data-protection
  rules.
- This project is a technical aid only. It does not replace the legally competent authority for
  issuing traffic fines, and no part of it should be used as sole evidence without human review.

## License

No licence file has been added yet, which means **all rights are reserved by default** — others
may not legally use, copy or distribute this code. If you want contributions, add a `LICENSE`
file (MIT or Apache-2.0 are common choices for projects like this).

---

📄 Vietnamese version of this guide: [docs/README.vi.md](docs/README.vi.md)
