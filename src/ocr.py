from __future__ import annotations

import argparse

import cv2

from pipeline import (
    choose_best_plate_read,
    draw_candidates,
    draw_status_lines,
    find_plate_candidates,
    open_source,
    read_frame,
    release_source,
    save_screenshots,
    set_tesseract_cmd,
    warp_plate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="License plate OCR stage.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--image", type=str, default=None, help="Optional image file for offline testing.")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
    parser.add_argument("--tesseract-cmd", type=str, default=None, help="Optional explicit path to the tesseract executable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_tesseract_cmd(args.tesseract_cmd)
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
            cleaned_text = ""
            detection_view = draw_candidates(frame, candidates)
            ocr_view = detection_view.copy()

            best_read = choose_best_plate_read(frame, candidates)
            if best_read is not None:
                aligned_plate = best_read.aligned_plate
                threshold = best_read.ocr_result.threshold_image
                cleaned_text = best_read.ocr_result.cleaned_text

            ocr_view = draw_status_lines(
                ocr_view,
                [
                    f"OCR: {cleaned_text or 'No text'}",
                    "Press s to save screenshots",
                    "Press q to quit",
                ],
            )

            cv2.imshow("OCR Stage", ocr_view)
            if aligned_plate is not None:
                cv2.imshow("Aligned Plate", aligned_plate)
            if threshold is not None:
                cv2.imshow("Thresholded Plate", threshold)

            key = cv2.waitKey(delay) & 0xFF
            if key == ord("s"):
                save_screenshots(
                    detection_view,
                    aligned_plate=aligned_plate,
                    ocr_frame=ocr_view,
                    threshold_image=threshold,
                )
                print("[saved] screenshots/detection.png, screenshots/alignment.png, screenshots/ocr.png")
            elif key == ord("q"):
                break
    finally:
        release_source(capture)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
