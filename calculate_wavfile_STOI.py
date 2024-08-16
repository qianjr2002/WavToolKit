import argparse
import numpy as np
import scipy.io.wavfile as wav
from pystoi.stoi import stoi

'''
https://github.com/mpariente/pystoi
'''

'''
python calculate_wavfile_STOI.py -c wav/clean/p232_005.wav -e wav/gtcrn_enh/p232_005.wav
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

def main():
    parser = argparse.ArgumentParser(description='Calculate STOI between clean and enhanced signal')
    parser.add_argument('--clean', '-c', required=True, help='Path to the clean signal WAV file')
    parser.add_argument('--enhanced', '-e', required=True, help='Path to the enhanced signal WAV file')
    args = parser.parse_args()

    stoi_value = calculate_stoi(args.clean, args.enhanced)
    print(f'STOI: {stoi_value}')

if __name__ == '__main__':
    main()