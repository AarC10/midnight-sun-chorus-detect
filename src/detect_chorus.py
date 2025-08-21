import sys
import os
import numpy as np
import librosa


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

if __name__ == '__main__':
    main()
