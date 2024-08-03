import argparse
import os
import numpy as np
from tqdm import tqdm
import scipy.io.wavfile as wav
from pystoi.stoi import stoi

'''
python calculate_STOI.py -c wav/clean/ -e wav/gtcrn_enh/ -p
'''

def calculate_stoi(clean_file, enhanced_file):
    fs_clean, clean_signal = wav.read(clean_file)
    fs_enhanced, enhanced_signal = wav.read(enhanced_file)

    if fs_clean != fs_enhanced:
        raise ValueError("Sampling rates of the two files do not match.")

    min_len = min(len(clean_signal), len(enhanced_signal))
    clean_signal = clean_signal[:min_len]
    enhanced_signal = enhanced_signal[:min_len]

    return stoi(clean_signal, enhanced_signal, fs_clean, extended=False)

def main():
    parser = argparse.ArgumentParser(description='Calculate average STOI between clean and enhanced signals')
    parser.add_argument('--clean_dir', '-c', required=True, help='Path to the directory containing clean signal WAV files')
    parser.add_argument('--enhanced_dir', '-e', required=True, help='Path to the directory containing enhanced signal WAV files')
    parser.add_argument('--show_progress', '-p', action='store_true', help='Show progress bar')
    parser.add_argument('--no_progress', '-np', default=True, action='store_false', help='Do not show progress bar')

    args = parser.parse_args()

    clean_files = [f for f in os.listdir(args.clean_dir) if f.endswith('.wav')]
    enhanced_files = [f for f in os.listdir(args.enhanced_dir) if f.endswith('.wav')]

    stoi_values = []
    iterator = tqdm(clean_files) if args.show_progress else clean_files

    for clean_file in iterator:
        if clean_file in enhanced_files:
            clean_path = os.path.join(args.clean_dir, clean_file)
            enhanced_path = os.path.join(args.enhanced_dir, clean_file)
            stoi_value = calculate_stoi(clean_path, enhanced_path)
            if not args.show_progress:
                print(f"File: {clean_file}, STOI score : {stoi_value}")
            stoi_values.append(stoi_value)
        else:
            print(f"Warning: {clean_file} not found in enhanced directory.")

    if stoi_values:
        average_stoi = np.mean(stoi_values)
        print(f'Average STOI: {average_stoi}')
    else:
        print('No matching files found.')

if __name__ == '__main__':
    main()
