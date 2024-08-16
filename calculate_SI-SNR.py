import os
import argparse
import torch
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
import soundfile as sf
from tqdm import tqdm

'''
python calculate_SI-SNR.py --clean_folder wav/clean --enhanced_folder wav/noisy
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

def calculate_average_si_snr(clean_folder, enhanced_folder):
    si_snr_list = []
    for clean_file in tqdm(os.listdir(clean_folder), desc='Average SI-SNR calculating'):
        if clean_file.endswith('.wav'):
            clean_path = os.path.join(clean_folder, clean_file)
            enhanced_path = os.path.join(enhanced_folder, clean_file)

            if not os.path.exists(enhanced_path):
                print(f"Enhanced file not found for: {clean_path}")
                continue

            si_snr_value = calculate_si_snr(enhanced_path, clean_path)
            si_snr_list.append(si_snr_value)
            
            # print(f"File: {clean_file}, SI-SNR: {si_snr_value}")

    if si_snr_list:
        average_si_snr = sum(si_snr_list) / len(si_snr_list)
        print("Average SI-SNR:", average_si_snr)
    else:
        print("No SI-SNR values calculated. Check if the folders contain matching WAV files.")

def main():
    parser = argparse.ArgumentParser(description="Calculate average SI-SNR for WAV files in specified folders.")
    parser.add_argument('--enhanced_folder','-e', type=str, required=True, help="Path to the folder containing enhanced WAV files")
    parser.add_argument('--clean_folder','-c', type=str, required=True, help="Path to the folder containing clean WAV files")
    args = parser.parse_args()
    
    calculate_average_si_snr(args.enhanced_folder, args.clean_folder)

if __name__ == "__main__":
    main()
