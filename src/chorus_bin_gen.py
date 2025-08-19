import os
import sys
import numpy as np
import librosa

def main():
    if len(sys.argv) < 2:
        print("Expected: python3 chorus_bin_gen.py <input.wav> <output.npy>")
        sys.exit(1)

    input_wav = sys.argv[1]
    output_npy = sys.argv[2]

    if not os.path.isfile(input_wav):
        print("Input wav file does not exist")
        sys.exit(1)

    hop_length = 512
    audio_time_series, sampling_rate = librosa.load(input_wav)
    chroma = librosa.feature.chroma_cqt(y=audio_time_series, sr=sampling_rate, hop_length=hop_length)
    chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)

    print(audio_time_series.shape)
    print("Sampling rate: ", sampling_rate)

    np.save(output_npy, chroma)
    print("Saved numpy template to " + output_npy)

if __name__ == '__main__':
    main()