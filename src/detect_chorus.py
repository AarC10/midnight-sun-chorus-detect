import sys
import os
import numpy as np
import librosa

# Tunables
DEFAULT_THRESHOLD = float(os.getenv("CHORUS_THRESHOLD", "0.8"))
MIN_RUN_SEC = float(os.getenv("CHORUS_MIN_RUN_SEC", "1.0"))  # min seconds above threshold to count
LIVE_COOLDOWN_SEC = float(os.getenv("CHORUS_LIVE_COOLDOWN_SEC", "5.0"))  # debouncer


def compute_chroma(audio: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
    chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    return chroma

def print_usage_and_exit():
    print("Usage:\n  Live:  python3 src/detect_chorus.py <chorus_template.npy>\n  File:  python3 src/detect_chorus.py <input.wav> <chorus_template.npy>")
    sys.exit(1)


def main():
    if len(sys.argv) == 2:
        chorus_npy = sys.argv[1]
        if not os.path.isfile(chorus_npy):
            print("Chorus template npy file does not exist")
            sys.exit(1)
        return
    if len(sys.argv) < 3:
        print_usage_and_exit()

    input_wav = sys.argv[1]
    chorus_npy = sys.argv[2]

    if not os.path.isfile(input_wav):
        print("Input wav file does not exist")
        sys.exit(1)
    if not os.path.isfile(chorus_npy):
        print("Chorus template npy file does not exist")
        sys.exit(1)

    hop_length = 512
    audio, sr = librosa.load(input_wav)
    chroma = compute_chroma(audio, sr, hop_length)


if __name__ == '__main__':
    main()
