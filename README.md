# Car Number Plate Extraction System

This project implements the three-step Automatic Number Plate Recognition (ANPR) pipeline described in *Car Number Plate Extraction in Three Steps: Detection, Alignment, and OCR* by Gabriel Baziramwabo. The system captures frames from a webcam, detects a plausible plate region with OpenCV, rectifies the plate with a perspective transform, extracts text with Tesseract OCR, validates the text against the Rwanda plate format, confirms the result across multiple observations, and saves trusted plates to a CSV log.

## Pipeline

1. Detection
   The frame is converted to grayscale, filtered, edge-enhanced, and searched for rectangular contours with a number-plate-like aspect ratio.
2. Alignment
   The best candidate is converted from a rotated quadrilateral into a fixed `450 x 140` plate image using a perspective warp.
3. OCR
   The aligned plate is thresholded and passed to Tesseract using a whitelist of uppercase letters and digits.
4. Validation
   The OCR result is normalized and checked against the expected Rwanda format: `AAA999A`.
5. Temporal confirmation
   The same valid plate must appear repeatedly before it is confirmed and written to the log.
6. Persistence
   Confirmed plates are appended to `data/plates.csv`.

## Project Structure

```text
car_plate_extraction/
├── README.md
├── requirements.txt
├── data/
│   └── plates.csv
├── screenshots/
└── src/
    ├── camera.py
    ├── detect.py
    ├── align.py
    ├── ocr.py
    ├── validate.py
    ├── main.py
    └── pipeline.py
```

## Installation

git clone https://github.com/humuraelvin/car-plate-extraction.git

This repository is configured for Python `3.11.9` because OpenCV and OCR dependencies are more stable there than on Python 3.13.

```bash
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
tesseract --version
```

If `tesseract` is not on `PATH`, pass its full path with `--tesseract-cmd`.

## Running the Stages

Camera check:

```bash
python src/camera.py
```

Detection only:

```bash
python src/detect.py
```

Detection + alignment:

```bash
python src/align.py
```

Detection + alignment + OCR:

```bash
python src/ocr.py
```

Detection + alignment + OCR + validation:

```bash
python src/validate.py
```

Full live pipeline with temporal confirmation and CSV logging:

```bash
python src/main.py
```

Useful options:

```bash
python src/main.py --camera 0 --width 1280 --height 720
python src/main.py --image path/to/test-image.jpg
python src/main.py --buffer-size 5 --min-confirmations 3 --cooldown 10
```

## Real-World Testing Workflow

1. Ask the vehicle owner for permission and explain the test is for a school project.
2. Start the full pipeline with `python src/main.py`.
3. Position the webcam so the plate fills a reasonable part of the frame.
4. Wait until the plate is shown as confirmed.
5. Press `s` to save:
   `screenshots/detection.png`, `screenshots/alignment.png`, and `screenshots/ocr.png`
6. Repeat on multiple vehicles.
7. Commit the generated screenshots and the updated CSV log to GitHub.

## Expected Plate Format

The validation stage expects plates in this format:

```text
AAA999A
```

Examples:

```text
RAH972U
RAB123A
```

The validator also applies light OCR correction for common confusions such as `O/0`, `S/5`, `B/8`, and `I/1`.

## Screenshots of Results

Run the system on real vehicles, press `s`, and commit the generated images so this section renders correctly on GitHub.

```markdown
![Detection result](screenshots/detection.png)
![Alignment result](screenshots/alignment.png)
![OCR result](screenshots/ocr.png)
```

## CSV Output

Confirmed plates are stored in `data/plates.csv` with this schema:

```csv
Plate Number,Timestamp
RAH972U,2026-03-12 10:45:16
```

## Submission Checklist

1. Test on multiple real vehicles.
2. Capture and commit the screenshots in `screenshots/`.
3. Push the project to GitHub.
4. Submit your name, class, and repository URL in the quiz form.
