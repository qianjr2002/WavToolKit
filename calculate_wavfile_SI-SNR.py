import os
import argparse
import torch
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
import soundfile as sf

'''
python calculate_si_snr.py --clean_file wav/clean/p232_001.wav --enhanced_file wav/enhanced/p232_001.wav
'''

'''
https://lightning.ai/docs/torchmetrics/stable/audio/scale_invariant_signal_noise_ratio.html
'''

def calculate_si_snr(enhanced_file, clean_file):
    enhanced_signal, _ = sf.read(enhanced_file)
    clean_signal, _ = sf.read(clean_file)

    min_length = min(len(clean_signal), len(enhanced_signal))
    clean_signal = clean_signal[:min_length]
    enhanced_signal = enhanced_signal[:min_length]

    enhanced_signal_tensor = torch.tensor(enhanced_signal)
    clean_signal_tensor = torch.tensor(clean_signal)

    si_snr_value = scale_invariant_signal_noise_ratio(enhanced_signal_tensor, clean_signal_tensor)
    return si_snr_value.item()

def main():
    parser = argparse.ArgumentParser(description="Calculate SI-SNR between enhanced and clean signals.")
    parser.add_argument('--enhanced_file', '-e', type=str, required=True, help="Path to the enhanced signal WAV file")
    parser.add_argument('--clean_file', '-c', type=str, required=True, help="Path to the clean signal WAV file")
    args = parser.parse_args()

    si_snr_value = calculate_si_snr(args.enhanced_file, args.clean_file)
    print("SI-SNR:", si_snr_value)

if __name__ == "__main__":
    main()
