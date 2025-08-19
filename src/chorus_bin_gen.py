import sys
import numpy as np
import librosa

def main():
    if len(sys.argv) < 2:
        print("Expected: python3 chorus_bin_gen.py <input.wav> <output.npy>")
        sys.exit(1)

    input_wav = sys.argv[1]
    output_npy = sys.argv[2]

    print(input_wav, output_npy)

if __name__ == '__main__':
    main()