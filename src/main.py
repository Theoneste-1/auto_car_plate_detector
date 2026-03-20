from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from temporal import TemporalPlateTracker
from pipeline import (
    CSV_PATH,
    SCREENSHOT_DIR,
    append_plate_log,
    choose_best_plate_read,
    draw_candidates,
    draw_status_lines,
    ensure_csv,
    find_plate_candidates,
    open_source,
    read_frame,
    release_source,
    save_screenshots,
    set_tesseract_cmd,
    warp_plate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full ANPR pipeline.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--image", type=str, default=None, help="Optional image file for offline testing.")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
    parser.add_argument("--buffer-size", type=int, default=5, help="Temporal buffer length.")
    parser.add_argument("--min-confirmations", type=int, default=3, help="Minimum repeated observations before confirmation.")
    parser.add_argument("--cooldown", type=float, default=10.0, help="Seconds before the same plate can be logged again.")
    parser.add_argument("--tesseract-cmd", type=str, default=None, help="Optional explicit path to the tesseract executable.")
    return parser.parse_args()


def get_next_test_folder(base_dir: Path) -> str:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("car_test_")]
    indices = [int(name.split("_")[-1]) for name in existing if name.split("_")[-1].isdigit()]
    next_idx = max(indices, default=0) + 1
    return f"car_test_{next_idx}"


def get_timestamp_folder() -> str:
    return f"auto_{time.strftime('%Y%m%d_%H%M%S')}"


def main() -> None:
    args = parse_args()
    set_tesseract_cmd(args.tesseract_cmd)
    ensure_csv(CSV_PATH)
    tracker = TemporalPlateTracker(
        buffer_size=args.buffer_size,
        min_confirmations=args.min_confirmations,
        cooldown_seconds=args.cooldown,
    )
    capture, image_frame = open_source(args.camera, args.image, args.width, args.height)
    delay = 0 if image_frame is not None else 1

    try:
        while True:
            ok, frame = read_frame(capture, image_frame)
            if not ok or frame is None:
                break

            candidates = find_plate_candidates(frame)
            aligned_plate = None
            threshold = None
            raw_text = ""
            valid_plate = None
            confirmed_plate = None
            logged_now = False
            detection_view = draw_candidates(frame, candidates)
            app_view = detection_view.copy()

            best_read = choose_best_plate_read(frame, candidates)
            if best_read is not None:
                aligned_plate = best_read.aligned_plate
                threshold = best_read.ocr_result.threshold_image
                raw_text = best_read.ocr_result.cleaned_text
                valid_plate = best_read.valid_plate
                if valid_plate:
                    confirmed_plate, logged_now = tracker.observe(valid_plate, now=time.time())
                    if logged_now and confirmed_plate:
                        append_plate_log(confirmed_plate, CSV_PATH)
                        folder = get_timestamp_folder()
                        save_screenshots(
                            detection_view,
                            aligned_plate=aligned_plate,
                            ocr_frame=app_view,
                            threshold_image=threshold,
                            sub_folder=folder,
                        )
                        print(f"[auto-save] {confirmed_plate} -> {CSV_PATH} and screenshots/{folder}")

            app_view = draw_status_lines(
                app_view,
                [
                    f"Detected candidates: {len(candidates)}",
                    f"OCR: {raw_text or 'No text'}",
                    f"Valid plate: {valid_plate or 'No match'}",
                    f"Confirmed plate: {confirmed_plate or 'Waiting for repeat observations'}",
                    "Press s to save screenshots | Press q to quit",
                ],
            )

            cv2.imshow("ANPR Pipeline", app_view)
            if aligned_plate is not None:
                cv2.imshow("Aligned Plate", aligned_plate)
            if threshold is not None:
                cv2.imshow("Thresholded Plate", threshold)

            key = cv2.waitKey(delay) & 0xFF
            if key == ord("s"):
                folder = get_next_test_folder(SCREENSHOT_DIR)
                save_screenshots(
                    detection_view,
                    aligned_plate=aligned_plate,
                    ocr_frame=app_view,
                    threshold_image=threshold,
                    sub_folder=folder,
                )
                # Manual override: always log to CSV if we have a valid plate or even just raw OCR
                log_text = valid_plate or raw_text or "MANUAL_SAVE_NO_OCR"
                append_plate_log(log_text, CSV_PATH)
                print(f"[manual-save] {log_text} -> {CSV_PATH} and screenshots/{folder}")
            elif key == ord("q"):
                break
    finally:
        release_source(capture)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
