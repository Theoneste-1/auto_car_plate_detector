import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pipeline
import pytesseract

def test_tesseract_discovery():
    print("Testing Tesseract discovery...")
    pipeline.set_tesseract_cmd(None)
    cmd = pytesseract.pytesseract.tesseract_cmd
    print(f"Current Tesseract cmd: {cmd}")
    
    if sys.platform.startswith("win"):
        if "tesseract.exe" in cmd.lower():
            print("SUCCESS: Tesseract path discovered or already set.")
        else:
            print("WARNING: Tesseract path not discovered. Is it installed in a standard location?")
    else:
        print(f"On {sys.platform}, assuming Tesseract is in PATH or handled by system.")

def test_camera_opening():
    print("\nTesting camera opening logic (will not fail if no camera found)...")
    try:
        # We don't want to actually block on camera index 0 if it pops up a window or something
        # but calling open_source should now use CAP_DSHOW on Windows.
        cap, frame = pipeline.open_source(camera_index=0)
        if cap is not None:
            print("SUCCESS: Camera opened successfully.")
            cap.release()
        else:
            print("INFO: Camera not opened (might be expected if no camera connected).")
    except Exception as e:
        print(f"INFO: Camera open failed as expected or due to: {e}")

if __name__ == "__main__":
    test_tesseract_discovery()
    test_camera_opening()
