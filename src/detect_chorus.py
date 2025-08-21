import sys
import os
import numpy as np
import librosa
import sounddevice as sd
import queue
import time

# Tunables
DEFAULT_THRESHOLD = float(os.getenv("CHORUS_THRESHOLD", "0.8"))
MIN_RUN_SEC = float(os.getenv("CHORUS_MIN_RUN_SEC", "1.0"))  # min seconds above threshold to count
LIVE_COOLDOWN_SEC = float(os.getenv("CHORUS_LIVE_COOLDOWN_SEC", "5.0"))  # debouncer
SMOOTH_SEC = float(os.getenv("CHORUS_SMOOTH_SEC", "0.0"))  # moving average window in seconds; 0 disables
SEGMENT_SIGNAL = os.getenv("CHORUS_SEGMENT_SIGNAL", "raw").lower()  # "smooth" or "raw"
START_THRESHOLD = float(os.getenv("CHORUS_START_THRESHOLD", str(DEFAULT_THRESHOLD)))
CONT_THRESHOLD = float(os.getenv("CHORUS_CONT_THRESHOLD", str(max(0.0, DEFAULT_THRESHOLD - 0.08))))
MAX_GAP_SEC = float(os.getenv("CHORUS_MAX_GAP_SEC", "0.25"))  # allow brief dips below cont threshold
ALIGN_TOL_SEC = float(os.getenv("CHORUS_ALIGN_TOL_SEC", "0.23"))  # tolerate ±sec frame misalignment
CHORUS_METHOD = os.getenv("CHORUS_METHOD", "dtw").lower()  # 'dtw' (default) or 'hyst'


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


def compute_similarities(chroma: np.ndarray, chorus_template: np.ndarray, frames_per_sec: float | None = None,
                         align_tol_sec: float = 0.0) -> np.ndarray:
    template_len = chorus_template.shape[1]
    if chroma.shape[1] < template_len:
        return np.empty(0, dtype=np.float32)

    # Precompute centered template
    t = chorus_template.astype(np.float32)
    # We'll compare on overlapping parts when shifting, so no single flattened template

    out = np.empty(chroma.shape[1] - template_len + 1, dtype=np.float32)

    max_shift_frames = 0
    if frames_per_sec is not None and align_tol_sec > 0:
        max_shift_frames = max(1, int(round(frames_per_sec * align_tol_sec)))

    for i in range(out.size):
        # Base window
        w = chroma[:, i:i + template_len].astype(np.float32)

        if max_shift_frames == 0:
            # No alignment search: straight centered-cosine over full window
            wf = w.flatten()
            tf = t.flatten()
            # center
            wf = wf - wf.mean()
            tf = tf - tf.mean()
            denom = (np.linalg.norm(wf) + 1e-9) * (np.linalg.norm(tf) + 1e-9)
            out[i] = float(np.dot(wf, tf) / denom)
            continue

        # Search over shifts in [-K..K] without wrap; compare only overlapping region
        best = -1.0
        T = template_len
        for s in range(-max_shift_frames, max_shift_frames + 1):
            if s == 0:
                w_slice = w
                t_slice = t
            elif s > 0:
                # shift template right by s -> compare w[:, s:] with t[:, :T-s]
                if s >= T:
                    continue
                w_slice = w[:, s:]
                t_slice = t[:, :T - s]
            else:  # s < 0
                ss = -s
                if ss >= T:
                    continue
                w_slice = w[:, :T - ss]
                t_slice = t[:, ss:]
            wf = w_slice.flatten()
            tf = t_slice.flatten()
            # center
            wf = wf - wf.mean()
            tf = tf - tf.mean()
            denom = (np.linalg.norm(wf) + 1e-9) * (np.linalg.norm(tf) + 1e-9)
            sim = float(np.dot(wf, tf) / denom)
            if sim > best:
                best = sim
        out[i] = best
    return out


def find_segments(similarities: np.ndarray, threshold: float, frames_per_sec: float, min_run_sec: float) -> list[
    tuple[int, int]]:
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


def find_segments_hysteresis(similarities: np.ndarray, frames_per_sec: float,
                             start_thresh: float, cont_thresh: float,
                             min_run_sec: float, max_gap_sec: float) -> list[tuple[int, int]]:
    if similarities.size == 0:
        return []
    min_run_frames = max(1, int(round(min_run_sec * frames_per_sec)))
    max_gap_frames = max(0, int(round(max_gap_sec * frames_per_sec)))

    segments: list[tuple[int, int]] = []
    in_run = False
    run_start = 0
    gap = 0

    for i, v in enumerate(similarities):
        if not in_run:
            if v >= start_thresh:
                in_run = True
                run_start = i
                gap = 0
        else:
            if v >= cont_thresh:
                gap = 0
            else:
                gap += 1
                if gap > max_gap_frames:
                    run_end = i - gap  # last frame before gap started
                    if run_end >= run_start and (run_end - run_start + 1) >= min_run_frames:
                        segments.append((run_start, run_end))
                    in_run = False
                    gap = 0
    # close trailing
    if in_run:
        run_end = len(similarities) - 1
        if (run_end - run_start + 1) >= min_run_frames:
            segments.append((run_start, run_end))
    return segments


def detect_dtw_segment(chroma_song: np.ndarray, chroma_tmpl: np.ndarray):
    # librosa expects shape (features, frames)
    # chroma already matches (12, T)
    x = chroma_tmpl.astype(np.float32)  # (12, n)
    y = chroma_song.astype(np.float32)  # (12, m)
    try:
        # subsequence alignment: match all of X inside Y
        d, wp = librosa.sequence.dtw(X=x, Y=y, metric='cosine', subseq=True)
    except Exception:
        # Fallback: regular DTW (may force alignment to start)
        d, wp = librosa.sequence.dtw(X=x, Y=y, metric='cosine', subseq=False)
    if wp is None or len(wp) == 0:
        return None

    path = np.array(wp)[::-1]
    i_idx = path[:, 0]
    j_idx = path[:, 1]
    n = int(x.shape[1])
    y_hits = np.full(n, -1, dtype=int)
    for i, j in zip(i_idx, j_idx):
        if 0 <= i < n and y_hits[i] == -1:
            y_hits[i] = j
    valid = y_hits[y_hits >= 0]
    if valid.size == 0:
        return None
    start_idx = int(valid.min())
    end_idx = int(valid.max())

    # Recompute cosine sim of 12D cols along mapped pairs
    sims = []
    for i in range(n):
        j = y_hits[i]
        if j >= 0:
            a = x[:, i]
            b = y[:, j]
            na = np.linalg.norm(a) + 1e-9
            nb = np.linalg.norm(b) + 1e-9
            sims.append(float(np.dot(a, b) / (na * nb)))
    score = float(np.mean(sims)) if sims else 0.0
    return start_idx, end_idx, score


def live_detect(chorus_npy):
    hop_length = 512
    sr = 22050  # default sampl rate
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
    print(
        "Usage:\n  Live:  python3 src/detect_chorus.py <chorus_template.npy>\n  File:  python3 src/detect_chorus.py <input.wav> <chorus_template.npy>")
    sys.exit(1)


def smooth_similarities(sim: np.ndarray, frames_per_sec: float, smooth_sec: float) -> np.ndarray:
    if smooth_sec <= 0 or sim.size == 0:
        return sim
    w = max(1, int(round(frames_per_sec * smooth_sec)))
    if w <= 1:
        return sim
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(sim, kernel, mode='same').astype(np.float32)

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

    frames_per_sec = sr / hop_length

    if CHORUS_METHOD == 'dtw':
        res = detect_dtw_segment(chroma, chorus_template)
        if not res:
            print("No chorus detected (DTW).")
            return
        start_idx, end_idx, score = res
        start_t = start_idx * hop_length / sr
        dur = max(0.0, (end_idx - start_idx + 1) / frames_per_sec)
        print(f"DTW best segment: {start_t:.2f}s (dur={dur:.2f}s, score={score:.2f})")
        return

    # Hysteresis/threshold method (default)
    raw_similarities = compute_similarities(chroma, chorus_template, frames_per_sec, ALIGN_TOL_SEC)
    similarities = smooth_similarities(raw_similarities, frames_per_sec, SMOOTH_SEC)

    # Select which signal to segment on
    seg_signal = similarities if SEGMENT_SIGNAL == "smooth" else raw_similarities

    # Report best match from raw
    if raw_similarities.size == 0:
        print("No chorus detected.")
        return
    best_idx = int(np.argmax(raw_similarities))
    best_time = best_idx * hop_length / sr
    best_raw = float(raw_similarities[best_idx])
    best_smooth = float(similarities[best_idx]) if similarities.size > best_idx else best_raw
    print(f"Best match at {best_time:.2f}s (raw={best_raw:.2f}, smooth={best_smooth:.2f})")

    # Segment detection using hysteresis
    segments = find_segments_hysteresis(
        seg_signal,
        frames_per_sec,
        START_THRESHOLD,
        CONT_THRESHOLD,
        MIN_RUN_SEC,
        MAX_GAP_SEC,
    )

    if not segments:
        if best_raw >= START_THRESHOLD:
            print(f"Chorus detected (peak-only) at ~{best_time:.2f}s (raw={best_raw:.2f})")
        return

    print("Chorus detected segments (start times in seconds):")
    for start_idx, end_idx in segments:
        t = start_idx * hop_length / sr
        peak = float(seg_signal[start_idx:end_idx + 1].max())
        dur = (end_idx - start_idx + 1) / frames_per_sec
        print(f"{t:.2f}s (peak={peak:.2f}, dur={dur:.2f}s)")


if __name__ == '__main__':
    main()
