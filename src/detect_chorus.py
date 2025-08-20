import numpy as np

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
TEMPLATE_PATH = "chorus_template.npy"
INPUT_DEVICE = None        # set to device index or None for default

def load_template(path: str) -> np.ndarray:
    template = np.load(path)
    template = template / (np.linalg.norm(template, axis=0, keepdims=True) + 1e-9)
    return template