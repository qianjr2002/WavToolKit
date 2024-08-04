import os
import argparse
import torch
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
import soundfile as sf

'''
python calculate_SI-SNR --clean_folder wav/clean --enhanced_folder wav/noisy
'''

def calculate_si_snr(clean_folder, noisy_folder):
    si_snr_list = []
    for clean_file in os.listdir(clean_folder):
        if clean_file.endswith('.wav'):

            clean_signal, sr = sf.read(os.path.join(clean_folder, clean_file))
            noisy_signal, sr = sf.read(os.path.join(noisy_folder, clean_file))  # 使用相同的文件名

            clean_signal_tensor = torch.tensor(clean_signal)
            noisy_signal_tensor = torch.tensor(noisy_signal)

            si_snr_value = scale_invariant_signal_noise_ratio(noisy_signal_tensor, clean_signal_tensor)
            si_snr_list.append(si_snr_value.item())
            
            print(f"File: {clean_file}, SI-SNR: {si_snr_value.item()}")

    average_sisnr = sum(si_snr_list) / len(si_snr_list)
    print("Average SI-SNR:", average_sisnr)

def main():
    parser = argparse.ArgumentParser(description="Calculate SI-SNR for WAV files in specified folders.")
    parser.add_argument('--clean_folder','-c', type=str, required=True, help="Path to the folder containing clean WAV files")
    parser.add_argument('--enhanced_folder','-e', type=str, required=True, help="Path to the folder containing enhanced WAV files")
    args = parser.parse_args()
    
    calculate_si_snr(args.clean_folder, args.noisy_folder)

if __name__ == "__main__":
    main()
