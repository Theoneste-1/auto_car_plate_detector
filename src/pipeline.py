from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract

MIN_AREA = 1200
AR_MIN = 2.0
AR_MAX = 8.0
RECTANGULARITY_MIN = 0.42
W_OUT = 450
H_OUT = 140
PLATE_RE = re.compile(r"^[A-Z]{3}[0-9]{3}[A-Z]$")
ALNUM_RE = re.compile(r"[^A-Z0-9]")
LETTER_TO_DIGIT = {
    "B": "8",
    "D": "0",
    "G": "6",
    "I": "1",
    "L": "1",
    "O": "0",
    "Q": "0",
    "S": "5",
    "Z": "2",
}
DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}
SCREENSHOT_DIR = Path("screenshots")
CSV_PATH = Path("data/plates.csv")


@dataclass(slots=True)
class PlateCandidate:
    rect: tuple
    contour: np.ndarray
    box: np.ndarray
    area: float
    aspect_ratio: float
    rectangularity: float
    score: float


@dataclass(slots=True)
class OCRResult:
    raw_text: str
    cleaned_text: str
    threshold_image: np.ndarray


@dataclass(slots=True)
class PlateReading:
    candidate: PlateCandidate
    aligned_plate: np.ndarray
    ocr_result: OCRResult
    valid_plate: str | None
    score: float


def set_tesseract_cmd(command: str | None) -> None:
    if command:
        pytesseract.pytesseract.tesseract_cmd = command


def open_source(
    camera_index: int = 0,
    image_path: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> tuple[cv2.VideoCapture | None, np.ndarray | None]:
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return None, frame

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap, None


def read_frame(
    capture: cv2.VideoCapture | None,
    image_frame: np.ndarray | None,
) -> tuple[bool, np.ndarray | None]:
    if image_frame is not None:
        return True, image_frame.copy()
    if capture is None:
        return False, None
    return capture.read()


def release_source(capture: cv2.VideoCapture | None) -> None:
    if capture is not None:
        capture.release()


def prepare_edge_map(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(filtered, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)


def find_plate_candidates(
    frame: np.ndarray,
    min_area: int = MIN_AREA,
    ar_min: float = AR_MIN,
    ar_max: float = AR_MAX,
    rectangularity_min: float = RECTANGULARITY_MIN,
) -> list[PlateCandidate]:
    edge_map = prepare_edge_map(frame)
    contours, _ = cv2.findContours(
        edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates: list[PlateCandidate] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(contour)
        (_, _), (w, h), _ = rect
        if w <= 1 or h <= 1:
            continue

        box_area = float(w * h)
        if box_area <= 0:
            continue

        aspect_ratio = max(w, h) / max(1.0, min(w, h))
        rectangularity = area / box_area
        if not (ar_min <= aspect_ratio <= ar_max):
            continue
        if rectangularity < rectangularity_min:
            continue

        box = cv2.boxPoints(rect).astype(np.int32)
        score = box_area * rectangularity
        candidates.append(
            PlateCandidate(
                rect=rect,
                contour=contour,
                box=box,
                area=area,
                aspect_ratio=aspect_ratio,
                rectangularity=rectangularity,
                score=score,
            )
        )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates


def draw_candidates(
    frame: np.ndarray,
    candidates: list[PlateCandidate],
    limit: int = 3,
) -> np.ndarray:
    canvas = frame.copy()
    for index, candidate in enumerate(candidates[:limit]):
        color = (0, 255, 0) if index == 0 else (0, 180, 255)
        label = "Plate" if index == 0 else f"Candidate {index + 1}"
        cv2.polylines(canvas, [candidate.box], True, color, 2)
        x, y = candidate.box[0]
        cv2.putText(
            canvas,
            label,
            (int(x), max(25, int(y) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return canvas


def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1)
    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(diffs)]
    bottom_left = pts[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def warp_plate(
    frame: np.ndarray,
    rect: tuple,
    output_size: tuple[int, int] = (W_OUT, H_OUT),
) -> np.ndarray:
    width, height = output_size
    src = order_points(cv2.boxPoints(rect))
    dst = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, transform, (width, height))


def preprocess_plate_for_ocr(plate_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    if float(np.mean(threshold)) < 127.0:
        threshold = cv2.bitwise_not(threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    return threshold


def ocr_plate(plate_img: np.ndarray) -> OCRResult:
    threshold = preprocess_plate_for_ocr(plate_img)
    config = (
        "--oem 3 --psm 7 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    raw_text = pytesseract.image_to_string(threshold, config=config)
    cleaned_text = ALNUM_RE.sub("", raw_text.upper())
    return OCRResult(raw_text=raw_text.strip(), cleaned_text=cleaned_text, threshold_image=threshold)


def normalize_plate_candidate(text: str) -> str | None:
    if len(text) != 7:
        return None

    chars = list(text)
    for index in (0, 1, 2, 6):
        chars[index] = DIGIT_TO_LETTER.get(chars[index], chars[index])
    for index in (3, 4, 5):
        chars[index] = LETTER_TO_DIGIT.get(chars[index], chars[index])

    candidate = "".join(chars)
    if PLATE_RE.fullmatch(candidate):
        return candidate
    return None


def extract_valid_plate(text: str) -> str | None:
    cleaned = ALNUM_RE.sub("", text.upper())
    if len(cleaned) < 7:
        return None

    windows = [cleaned] if len(cleaned) == 7 else [cleaned[i : i + 7] for i in range(len(cleaned) - 6)]
    for window in windows:
        if PLATE_RE.fullmatch(window):
            return window
        normalized = normalize_plate_candidate(window)
        if normalized:
            return normalized
    return None


def choose_best_plate_read(
    frame: np.ndarray,
    candidates: list[PlateCandidate],
    limit: int = 5,
) -> PlateReading | None:
    best_read: PlateReading | None = None
    for candidate in candidates[:limit]:
        aligned_plate = warp_plate(frame, candidate.rect)
        ocr_result = ocr_plate(aligned_plate)
        valid_plate = extract_valid_plate(ocr_result.cleaned_text)
        score = float(len(ocr_result.cleaned_text))
        if valid_plate:
            score += 100.0
        score += candidate.rectangularity * 10.0

        current_read = PlateReading(
            candidate=candidate,
            aligned_plate=aligned_plate,
            ocr_result=ocr_result,
            valid_plate=valid_plate,
            score=score,
        )
        if best_read is None or current_read.score > best_read.score:
            best_read = current_read
    return best_read


def ensure_csv(csv_path: Path = CSV_PATH) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Plate Number", "Timestamp"])
    return csv_path


def append_plate_log(
    plate: str,
    csv_path: Path = CSV_PATH,
    timestamp: str | None = None,
) -> None:
    csv_path = ensure_csv(csv_path)
    stamp = timestamp or time.strftime("%Y-%m-%d %H:%M:%S")
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([plate, stamp])


def save_screenshots(
    detection_frame: np.ndarray,
    aligned_plate: np.ndarray | None = None,
    ocr_frame: np.ndarray | None = None,
    threshold_image: np.ndarray | None = None,
    base_directory: Path = SCREENSHOT_DIR,
    sub_folder: str | None = None,
) -> Path:
    target_dir = base_directory if sub_folder is None else base_directory / sub_folder
    target_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target_dir / "detection.png"), detection_frame)
    if aligned_plate is not None:
        cv2.imwrite(str(target_dir / "alignment.png"), aligned_plate)
    if ocr_frame is not None:
        cv2.imwrite(str(target_dir / "ocr.png"), ocr_frame)
    if threshold_image is not None:
        cv2.imwrite(str(target_dir / "ocr_threshold.png"), threshold_image)
    return target_dir


def draw_status_lines(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = frame.copy()
    origin_y = 30
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (20, origin_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
        )
        cv2.putText(
            canvas,
            line,
            (20, origin_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        origin_y += 30
    return canvas
