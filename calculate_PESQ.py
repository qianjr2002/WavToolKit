import argparse
import os
from scipy.io import wavfile
from pesq import pesq

'''
python calculate_PESQ.py --clean_folder wav/clean --enhanced_folder wav/noisy --mode wb
python calculate_PESQ.py --clean_folder wav/clean --enhanced_folder wav/noisy
'''

def calculate_pesq_scores(clean_folder, enhanced_folder, mode):
    clean_files = [os.path.join(clean_folder, f) for f in os.listdir(clean_folder) if f.endswith(".wav")]
    
    pesq_scores = []
    
    for clean_file in clean_files:
        clean_filename = os.path.basename(clean_file)
        enhanced_file = os.path.join(enhanced_folder, clean_filename)
        if os.path.exists(enhanced_file):
            rate_clean, ref = wavfile.read(clean_file)
            _, deg = wavfile.read(enhanced_file)
            pesq_score = pesq(rate_clean, ref, deg, mode)
            print(f"File: {clean_filename}, PESQ score ({mode}): {pesq_score}")
            pesq_scores.append(pesq_score)
        else:
            print(f"File: {clean_filename}, No corresponding enhanced file found")
    
    average_score = sum(pesq_scores) / len(pesq_scores) if pesq_scores else 0
    print(f"Average PESQ score ({mode}): {average_score}")

def main():
    parser = argparse.ArgumentParser(description="Calculate PESQ scores for WAV files in specified folders.")
    parser.add_argument('--clean_folder', '-c', type=str, required=True, help="Path to the folder containing clean WAV files")
    parser.add_argument('--enhanced_folder', '-e', type=str, required=True, help="Path to the folder containing enhanced WAV files")
    parser.add_argument('--mode', '-m', type=str, choices=['wb', 'nb'], default='wb', help="PESQ mode: 'wb' for wideband or 'nb' for narrowband (default: 'wb')")
    args = parser.parse_args()
    
    calculate_pesq_scores(args.clean_folder, args.enhanced_folder, args.mode)

if __name__ == "__main__":
    main()
