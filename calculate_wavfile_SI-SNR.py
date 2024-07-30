import argparse
import torch
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
import soundfile as sf

'''
python calculate_si_snr.py --clean_file wav/clean/p232_001.wav --noisy_file wav/noisy/p232_001.wav
'''

def calculate_si_snr(clean_file, noisy_file):
    clean_signal, sr = sf.read(clean_file)
    noisy_signal, sr = sf.read(noisy_file)
    
    clean_signal_tensor = torch.tensor(clean_signal)
    noisy_signal_tensor = torch.tensor(noisy_signal)
    
    si_snr_value = scale_invariant_signal_noise_ratio(noisy_signal_tensor, clean_signal_tensor)
    return si_snr_value.item()

def main():
    parser = argparse.ArgumentParser(description="Calculate SI-SNR between clean and noisy signals.")
    parser.add_argument('--clean_file', type=str, required=True, help="Path to the clean signal WAV file")
    parser.add_argument('--noisy_file', type=str, required=True, help="Path to the noisy signal WAV file")
    args = parser.parse_args()
    
    si_snr_value = calculate_si_snr(args.clean_file, args.noisy_file)
    print("SI-SNR:", si_snr_value)

if __name__ == "__main__":
    main()
