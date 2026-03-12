from __future__ import annotations

import argparse

import cv2

from pipeline import open_source, read_frame, release_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera sanity check for the ANPR project.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open.")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture, image_frame = open_source(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
    )
    try:
        while True:
            ok, frame = read_frame(capture, image_frame)
            if not ok or frame is None:
                break

            cv2.putText(
                frame,
                "Camera OK - press q to quit",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Camera Test", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        release_source(capture)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
