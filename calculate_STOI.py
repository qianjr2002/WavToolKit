import argparse
import os
import numpy as np
from tqdm import tqdm
import scipy.io.wavfile as wav
from pystoi.stoi import stoi

'''
python calculate_STOI.py -c wav/clean/ -e wav/gtcrn_enh/
'''

def calculate_stoi(clean_file, enhanced_file):
    fs_clean, clean_signal = wav.read(clean_file)
    fs_enhanced, enhanced_signal = wav.read(enhanced_file)
    
    if fs_clean != fs_enhanced:
        raise ValueError("Sampling rates of the two files do not match.")
    
    min_len = min(len(clean_signal), len(enhanced_signal))
    clean_signal = clean_signal[:min_len]
    enhanced_signal = enhanced_signal[:min_len]
    
    stoi_value = stoi(clean_signal, enhanced_signal, fs_clean, extended=False)
    return stoi_value

def calculate_average_stoi(clean_folder, enhanced_folder):
    stoi_list = []
    for clean_file in tqdm(os.listdir(clean_folder), desc='Average STOI calculating'):
        if clean_file.endswith('.wav'):
            clean_path = os.path.join(clean_folder, clean_file)
            enhanced_path = os.path.join(enhanced_folder, clean_file)

            if not os.path.exists(enhanced_path):
                print(f"Enhanced file not found for: {clean_path}")
                continue

            stoi_value = calculate_stoi(clean_path, enhanced_path)
            stoi_list.append(stoi_value)
            
            # print(f"File: {clean_file}, STOI: {stoi_value}")

    if stoi_list:
        average_stoi = sum(stoi_list) / len(stoi_list)
        print("Average STOI:", average_stoi)
    else:
        print("No STOI values calculated. Check if the folders contain matching WAV files.")

def main():
    parser = argparse.ArgumentParser(description="Calculate average STOI for WAV files in specified folders.")
    parser.add_argument('--clean_folder', '-c', type=str, required=True, help="Path to the folder containing clean WAV files")
    parser.add_argument('--enhanced_folder', '-e', type=str, required=True, help="Path to the folder containing enhanced WAV files")
    args = parser.parse_args()
    
    calculate_average_stoi(args.clean_folder, args.enhanced_folder)

if __name__ == "__main__":
    main()
