import argparse
import numpy as np
import scipy.io.wavfile as wav
from pystoi.stoi import stoi

'''
https://github.com/mpariente/pystoi
'''

'''
STOI:
python calculate_wavfile_STOI.py -c wav/clean/p232_005.wav -e wav/gtcrn_enh/p232_005.wav
eSTOI:
python calculate_wavfile_STOI.py -c wav/clean/p232_005.wav -e wav/gtcrn_enh/p232_005.wav --extended
'''

def calculate_stoi(clean_file, enhanced_file, extended):
    fs_clean, clean_signal = wav.read(clean_file)
    fs_enhanced, enhanced_signal = wav.read(enhanced_file)
    
    if fs_clean != fs_enhanced:
        raise ValueError("Sampling rates of the two files do not match.")
    
    min_len = min(len(clean_signal), len(enhanced_signal))
    clean_signal = clean_signal[:min_len]
    enhanced_signal = enhanced_signal[:min_len]
    
    stoi_value = stoi(clean_signal, enhanced_signal, fs_clean, extended=extended)
    return stoi_value

def main():
    parser = argparse.ArgumentParser(description='Calculate STOI or eSTOI between clean and enhanced signal')
    parser.add_argument('--clean', '-c', required=True, help='Path to the clean signal WAV file')
    parser.add_argument('--enhanced', '-e', required=True, help='Path to the enhanced signal WAV file')
    parser.add_argument('--extended', action='store_true', help='Calculate eSTOI if this flag is set (default: False for STOI)')
    args = parser.parse_args()

    # extended will be True if the flag is set, otherwise False
    stoi_value = calculate_stoi(args.clean, args.enhanced, args.extended)
    
    if args.extended:
        print(f'eSTOI: {stoi_value}')
    else:
        print(f'STOI: {stoi_value}')

if __name__ == '__main__':
    main()