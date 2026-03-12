from __future__ import annotations

import argparse

import cv2

from pipeline import (
    draw_candidates,
    draw_status_lines,
    find_plate_candidates,
    open_source,
    read_frame,
    release_source,
    save_screenshots,
    warp_plate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="License plate alignment stage.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--image", type=str, default=None, help="Optional image file for offline testing.")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture, image_frame = open_source(args.camera, args.image, args.width, args.height)
    delay = 0 if image_frame is not None else 1

    try:
        while True:
            ok, frame = read_frame(capture, image_frame)
            if not ok or frame is None:
                break

            candidates = find_plate_candidates(frame)
            aligned_plate = None
            alignment_view = draw_candidates(frame, candidates)
            if candidates:
                aligned_plate = warp_plate(frame, candidates[0].rect)

            alignment_view = draw_status_lines(
                alignment_view,
                [
                    "Alignment stage",
                    "Press s to save detection.png and alignment.png",
                    "Press q to quit",
                ],
            )

            cv2.imshow("Plate Alignment", alignment_view)
            if aligned_plate is not None:
                cv2.imshow("Aligned Plate", aligned_plate)

            key = cv2.waitKey(delay) & 0xFF
            if key == ord("s"):
                save_screenshots(alignment_view, aligned_plate=aligned_plate)
                print("[saved] screenshots/detection.png and screenshots/alignment.png")
            elif key == ord("q"):
                break
    finally:
        release_source(capture)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
