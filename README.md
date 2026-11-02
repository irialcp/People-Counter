# People Counter

A Python computer vision project for detecting people in videos and webcam streams using YOLOv8 and OpenCV.

The repository includes two scripts: one for person detection in a video file, and another for counting people currently visible in a webcam stream, with configurable image effects to experiment with detection under altered visual conditions.

## Features

- Person detection using YOLOv8.
- Bounding box visualization.
- Video file and webcam input.
- Per-frame person counting in webcam mode.
- Configurable Gaussian noise and color quantization applied before detection.
- Interactive adjustment of image effect parameters.
- Two configurable polygonal overlays in video mode.

> This project is an experimental prototype. It does not yet implement persistent identity tracking or entry and exit counting.

## Tech Stack

- Python
- Ultralytics YOLOv8
- OpenCV
- NumPy
- pandas

## Repository Structure

```text
People-Counter/
├── README.md        # Project documentation
├── main.py          # Person detection in a video file
├── test.py          # Webcam detection, counting, and configurable effects
├── coco.txt         # Class names used by main.py
├── download.txt     # Link to an external resource
└── yolov8s.pt       # Model weights used by main.py
```

Additional resources required by the scripts:

- `peoplecount1.mp4`: the input video expected by `main.py`; not included in the repository.
- `yolov8n.pt`: the model used by `test.py`; not included in the repository.

## Installation

### 1. Clone the repository

```bash
git clone [https://github.com/irialcp/People-Counter.git](https://github.com/irialcp/People-Counter.git)
cd People-Counter
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows, using PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python numpy pandas
```

The repository does not currently include a `requirements.txt` file with pinned dependency versions.

## Usage

Run all commands from the project's root directory, as the scripts use relative paths to load files.

### Webcam Mode

Run:

```bash
python test.py
```

The script uses the default webcam:

```python
cap = cv2.VideoCapture(0)
```

The configured model is:

```python
model = YOLO('yolov8n.pt')
```

Make sure the model weights are available or that your environment allows Ultralytics to retrieve them.

At startup, the script asks for three parameters:

| Parameter | Range | Description |
|-----------|-------|-------------|
| Gaussian noise | 0–100 | Standard deviation of the noise added to the image. Set to `0` to disable noise. |
| Opacity | 0.0–1.0 | Available in the interface, but currently has no visual effect. |
| Quantization levels | 2–256 | Number of intensity levels per color channel. Lower values produce stronger quantization. |

To start without intentional image degradation, use:

```text
Noise: 0
Opacity: 1
Quantization levels: 256
```

Frames are processed and resized to `800 × 800` pixels before detection.

The display shows:

- Bounding boxes around detected people.
- Numeric labels associated with detections in the current frame.
- The number of people currently detected.
- The selected effect parameters.

> The current terminal prompts and on-screen labels are in Italian.

#### Controls

| Key | Action |
|-----|--------|
| `Esc` | Exit the application. |
| `r` | Enter new effect parameters in the terminal. |

After pressing `r`, return to the terminal to enter the new values.

### Video Mode

Place a video named `peoplecount1.mp4` in the project's root directory, then run:

```bash
python main.py
```

To use a different video, edit this line in `main.py`:

```python
cap = cv2.VideoCapture('peoplecount1.mp4')
```

The script:

1. Loads the `yolov8s.pt` model.
2. Reads class names from `coco.txt`.
3. Processes every second frame.
4. Resizes frames to `1020 × 500` pixels.
5. Draws bounding boxes around detected people.
6. Displays two polygonal overlays.

Press `Esc` to exit.

Moving the mouse over the window prints the pointer coordinates to the terminal, which can help when configuring the polygon vertices.

## Configuring the Areas

In `main.py`, the two areas are defined by the following coordinates:

```python
area1 = [(312, 388), (289, 390), (474, 469), (497, 462)]
area2 = [(279, 392), (250, 397), (423, 477), (454, 469)]
```

These coordinates refer to the resized `1020 × 500` image.

Adjust them to match the scene in your video.

> The areas are currently visual overlays only. They are not used to detect crossings or update entry and exit counters.

## Webcam Processing Pipeline

```text
Webcam capture
      ↓
Gaussian noise
      ↓
Color quantization
      ↓
Opacity function
(currently has no effect)
      ↓
Resizing and color conversion
      ↓
YOLOv8 detection
      ↓
Person-class filtering
      ↓
Counting and visualization
```

This mode supports qualitative exploration of how noise and color quantization affect detection. It does not include automated accuracy evaluation.

## Current Limitations

### Counting and Identity Tracking

The IDs displayed by `test.py` are based on detection indices within each frame. They are not persistent identities across frames.

As a result:

- The `TOTALE` value represents the number of people detected in the current frame.
- It is not a cumulative count of unique people.
- IDs may change between frames.
- Entries, exits, and individual trajectories cannot be reliably reconstructed.

### Opacity Effect

The `add_opacity()` function blends two identical copies of the same image. Changing `alpha` therefore does not currently produce a visual change.

### Other Limitations

- No automated tests are included: `test.py` is an interactive application.
- Results, statistics, and annotated videos are not exported.
- Resizing frames to fixed dimensions may distort the original aspect ratio.
- Performance and detection quality depend on hardware, scene conditions, and effect parameters.
- The application requires an environment that supports OpenCV GUI windows. Webcam mode also requires camera access.

## Potential Improvements

- Integrate a tracker with persistent object IDs.
- Implement area-crossing detection.
- Add separate entry and exit counters.
- Fix or replace the opacity effect.
- Add command-line arguments for the input source, model, and effect parameters.
- Export statistics to CSV.
- Save annotated video output.
- Measure FPS and inference time.
- Add a dependency file with verified package versions.
- Compare detection performance on original and degraded images.

## License

The repository does not currently include a `LICENSE` file.
