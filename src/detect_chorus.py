import sys
import os
import numpy as np
import librosa


# Tunables
DEFAULT_THRESHOLD = float(os.getenv("CHORUS_THRESHOLD", "0.5"))
MIN_RUN_SEC = float(os.getenv("CHORUS_MIN_RUN_SEC", "1.0"))  # min seconds above threshold to count
LIVE_COOLDOWN_SEC = float(os.getenv("CHORUS_LIVE_COOLDOWN_SEC", "2.0"))  # debouncer


def compute_chroma(audio: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
    chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    return chroma


def load_template(path: str) -> np.ndarray:
    try:
        t = np.load(path)
    except Exception as e:
        print("Failed to load chorus template .npy. Make sure you pass the template file. Error:", e)
        sys.exit(1)
    if t.ndim != 2 or t.shape[0] != 12:
        print("Invalid template shape. Expected (12, T) chroma template.")
        sys.exit(1)
    return t.astype(np.float32)





def compute_similarities(chroma: np.ndarray, chorus_template: np.ndarray) -> np.ndarray:
    template_len = chorus_template.shape[1]
    if chroma.shape[1] < template_len:
        return np.empty(0, dtype=np.float32)
    t_flat = chorus_template.flatten().astype(np.float32)
    t_flat = t_flat - t_flat.mean()
    t_norm = np.linalg.norm(t_flat) + 1e-9

    out = np.empty(chroma.shape[1] - template_len + 1, dtype=np.float32)
    for i in range(out.size):
        window = chroma[:, i:i+template_len]
        w_flat = window.flatten().astype(np.float32)
        w_flat = w_flat - w_flat.mean()
        denom = (np.linalg.norm(w_flat) + 1e-9) * t_norm
        out[i] = float(np.dot(w_flat, t_flat) / denom)
    return out


def find_segments(similarities: np.ndarray, threshold: float, frames_per_sec: float, min_run_sec: float) -> list[tuple[int, int]]:
    if similarities.size == 0:
        return []
    min_run_frames = max(3, int(min_run_sec * frames_per_sec))
    idxs = np.where(similarities > threshold)[0]
    if idxs.size == 0:
        return []

    segments: list[tuple[int, int]] = []
    run_start = idxs[0]
    prev = idxs[0]
    for idx in idxs[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        if (prev - run_start + 1) >= min_run_frames:
            segments.append((run_start, prev))
        run_start = idx
        prev = idx
    if (prev - run_start + 1) >= min_run_frames:
        segments.append((run_start, prev))
    return segments

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

    chorus_template = load_template(chorus_npy)

    similarities = compute_similarities(chroma, chorus_template)

    threshold = DEFAULT_THRESHOLD

    if similarities.size == 0:
        print("No chorus detected.")
        return

    best_idx = int(np.argmax(similarities))
    best_time = best_idx * hop_length / sr
    print(f"Best match at {best_time:.2f}s (similarity={similarities[best_idx]:.2f})")

    frames_per_sec = sr / hop_length
    segments = find_segments(similarities, threshold, frames_per_sec, MIN_RUN_SEC)

    if not segments:
        if similarities[best_idx] >= threshold:
            print(f"Chorus detected (peak-only) at ~{best_time:.2f}s (similarity={similarities[best_idx]:.2f})")
        else:
            return

    print("Chorus detected segments (start times in seconds):")
    for start_idx, end_idx in segments:
        t = start_idx * hop_length / sr
        peak = float(similarities[start_idx:end_idx+1].max())
        dur = (end_idx - start_idx + 1) / frames_per_sec
        print(f"{t:.2f}s (peak={peak:.2f}, dur={dur:.2f}s)")

if __name__ == '__main__':
    main()
