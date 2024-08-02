import argparse
import os
from scipy.io import wavfile
from pesq import pesq

'''
python calculate_PESQ.py --clean_folder wav/clean --enhanced_folder wav/noisy --mode wb
python calculate_PESQ.py --clean_folder wav/clean --enhanced_folder wav/noisy
'''

def calculate_pesq_scores(clean_folder, noisy_folder, mode):
    clean_files = [os.path.join(clean_folder, f) for f in os.listdir(clean_folder) if f.endswith(".wav")]
    
    pesq_scores = []
    
    for clean_file in clean_files:
        clean_filename = os.path.basename(clean_file)
        noisy_file = os.path.join(noisy_folder, clean_filename)
        if os.path.exists(noisy_file):
            rate_clean, ref = wavfile.read(clean_file)
            _, deg = wavfile.read(noisy_file)
            pesq_score = pesq(rate_clean, ref, deg, mode)
            print(f"File: {clean_filename}, PESQ score ({mode}): {pesq_score}")
            pesq_scores.append(pesq_score)
        else:
            print(f"File: {clean_filename}, No corresponding noisy file found")
    
    average_score = sum(pesq_scores) / len(pesq_scores) if pesq_scores else 0
    print(f"Average PESQ score ({mode}): {average_score}")

def main():
    parser = argparse.ArgumentParser(description="Calculate PESQ scores for WAV files in specified folders.")
    parser.add_argument('--clean_folder', type=str, required=True, help="Path to the folder containing clean WAV files")
    parser.add_argument('--enhanced_folder', type=str, required=True, help="Path to the folder containing enhanced WAV files")
    parser.add_argument('--mode', type=str, choices=['wb', 'nb'], default='wb', help="PESQ mode: 'wb' for wideband or 'nb' for narrowband (default: 'wb')")
    args = parser.parse_args()
    
    calculate_pesq_scores(args.clean_folder, args.noisy_folder, args.mode)

if __name__ == "__main__":
    main()
