import argparse
from scipy.io import wavfile
from pesq import pesq

'''
python calculate_wavfile_PESQ.py --clean_file wav/clean/p232_001.wav --noisy_file wav/noisy/p232_001.wav --mode wb
python calculate_wavfile_PESQ.py --clean_file wav/clean/p232_001.wav --noisy_file wav/noisy/p232_001.wav
'''

def calculate_pesq(clean_file, noisy_file, mode):
    rate, ref = wavfile.read(clean_file)
    rate, deg = wavfile.read(noisy_file)
    pesq_score = pesq(rate, ref, deg, mode)
    return pesq_score

def main():
    parser = argparse.ArgumentParser(description="Calculate PESQ for WAV files.")
    parser.add_argument('--clean_file', type=str, required=True, help="Path to the clean WAV file")
    parser.add_argument('--noisy_file', type=str, required=True, help="Path to the noisy WAV file")
    parser.add_argument('--mode', type=str, choices=['wb', 'nb'], default='wb', help="PESQ mode: 'wb' for wideband or 'nb' for narrowband (default: 'wb')")
    args = parser.parse_args()
    
    pesq_score = calculate_pesq(args.clean_file, args.noisy_file, args.mode)
    print(f"PESQ score ({args.mode}):", pesq_score)

if __name__ == "__main__":
    main()
