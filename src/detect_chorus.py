import numpy as np
import sys

# Config
SR = 22050 # derived from generator script
HOP = 512
BLOCK_SECONDS = 3.0        # analysis window size
STEP_SECONDS  = 0.75       # slide step
THRESHOLD = 0.78           # DTW score/length (lower==better)
REQUIRED_HITS = 2          # consecutive windows below threshold
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 921600
SERIAL_LINE = b"brightness 7000\n"
DEBOUNCE_SECONDS = 7.0     # don't re-trigger faster than this
TEMPLATE_PATH = "output.npy"
INPUT_DEVICE = None        # set to device index or None for default

def load_template(path: str) -> np.ndarray:
    template = np.load(path)
    template = template / (np.linalg.norm(template, axis=0, keepdims=True) + 1e-9)
    return template

def main():
    try:
        template = load_template(TEMPLATE_PATH)
        print(template)
    except Exception as e:
        print("Failed to load template:", e)
        sys.exit(1)


if __name__ == '__main__':
    main()