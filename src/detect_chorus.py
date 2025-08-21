import sys
import os
import numpy as np
import librosa
import sounddevice as sd
import queue
import time

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


def cosine_sim_centered(a_flat: np.ndarray, b_flat: np.ndarray) -> float:
    a = a_flat - a_flat.mean()
    b = b_flat - b_flat.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def update_chroma_buffer(buf: np.ndarray, new_frames: np.ndarray) -> np.ndarray:
    shape = buf.shape[1]
    n = new_frames.shape[1]
    if n >= shape:
        return new_frames[:, -shape:]
    buf = np.roll(buf, -n, axis=1)
    buf[:, -n:] = new_frames
    return buf


def compute_similarities(chroma: np.ndarray, chorus_template: np.ndarray) -> np.ndarray:
    template_len = chorus_template.shape[1]
    if chroma.shape[1] < template_len:
        return np.empty(0, dtype=np.float32)
    temp_flat = chorus_template.flatten().astype(np.float32)
    temp_flat = temp_flat - temp_flat.mean()
    t_norm = np.linalg.norm(temp_flat) + 1e-9

    out = np.empty(chroma.shape[1] - template_len + 1, dtype=np.float32)
    for i in range(out.size):
        window = chroma[:, i:i+template_len]
        win_flat = window.flatten().astype(np.float32)
        win_flat = win_flat - win_flat.mean()
        denom = (np.linalg.norm(win_flat) + 1e-9) * t_norm
        out[i] = float(np.dot(win_flat, temp_flat) / denom)
    return out


def find_segments(similarities: np.ndarray, threshold: float, frames_per_sec: float, min_run_sec: float) -> list[tuple[int, int]]:
    if similarities.size == 0:
        return []
    min_run_frames = max(3, int(min_run_sec * frames_per_sec))
    indices = np.where(similarities > threshold)[0]
    if indices.size == 0:
        return []

    segments: list[tuple[int, int]] = []
    run_start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
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


def live_detect(chorus_npy):
    hop_length = 512
    sr = 22050 # default sampl rate
    blocksize = hop_length * 20  # ~0.46s per block
    chorus_template = load_template(chorus_npy)
    template_len = chorus_template.shape[1]
    chroma_buffer = np.zeros((12, template_len), dtype=np.float32)
    q_audio = queue.Queue()
    last_detect_time = -1e9

    def audio_callback(indata, frames, time_info, status):
        q_audio.put(indata.copy())

    try:
        stream = sd.InputStream(
            channels=1, samplerate=sr, blocksize=blocksize, callback=audio_callback
        )
    except Exception as e:
        print("Could not open audio input device. Check your microphone and permissions. Error:", e)
        sys.exit(1)

    print("Listening for chorus... Press Ctrl+C to stop.")
    with stream:
        try:
            while True:
                audio_chunk = q_audio.get().flatten()
                chroma = compute_chroma(audio_chunk, sr, hop_length)
                chroma_buffer = update_chroma_buffer(chroma_buffer, chroma)
                sim = cosine_sim_centered(chroma_buffer.flatten(), chorus_template.flatten())
                now = time.time()
                if sim > DEFAULT_THRESHOLD and (now - last_detect_time) > LIVE_COOLDOWN_SEC:
                    print(f"Chorus detected (live) (similarity={sim:.2f})")
                    last_detect_time = now
        except KeyboardInterrupt:
            print("Stopped.")


def print_usage_and_exit():
    print("Usage:\n  Live:  python3 src/detect_chorus.py <chorus_template.npy>\n  File:  python3 src/detect_chorus.py <input.wav> <chorus_template.npy>")
    sys.exit(1)


def main():
    if len(sys.argv) == 2:
        chorus_npy = sys.argv[1]
        if not os.path.isfile(chorus_npy):
            print("Chorus template npy file does not exist")
            sys.exit(1)
        live_detect(chorus_npy)
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
