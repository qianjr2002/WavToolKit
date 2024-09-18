import os
import argparse
import torch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
import soundfile as sf
from tqdm import tqdm

'''
python calculate_SI-SDR.py --clean_folder wav/clean --enhanced_folder wav/gtcrn_enh
'''

def calculate_si_sdr(enhanced_file, clean_file):
    enhanced_signal, _ = sf.read(enhanced_file)
    clean_signal, _ = sf.read(clean_file)

    min_length = min(len(clean_signal), len(enhanced_signal))
    clean_signal = clean_signal[:min_length]
    enhanced_signal = enhanced_signal[:min_length]

    enhanced_signal_tensor = torch.tensor(enhanced_signal)
    clean_signal_tensor = torch.tensor(clean_signal)
    si_sdr = ScaleInvariantSignalDistortionRatio()
    si_sdr_value = si_sdr(enhanced_signal_tensor, clean_signal_tensor)
    return si_sdr_value.item()

def calculate_average_si_sdr(clean_folder, enhanced_folder):
    si_sdr_list = []
    for clean_file in tqdm(os.listdir(clean_folder), desc='Average SI-SDR calculating'):
        if clean_file.endswith('.wav'):
            clean_path = os.path.join(clean_folder, clean_file)
            enhanced_path = os.path.join(enhanced_folder, clean_file)

            if not os.path.exists(enhanced_path):
                print(f"Enhanced file not found for: {clean_path}")
                continue

            si_sdr_value = calculate_si_sdr(enhanced_path, clean_path)
            si_sdr_list.append(si_sdr_value)
            
            # print(f"File: {clean_file}, SI-SDR: {si_sdr_value}")

    if si_sdr_list:
        average_si_sdr = sum(si_sdr_list) / len(si_sdr_list)
        print("Average SI-SDR:", average_si_sdr)
    else:
        print("No SI-SDR values calculated. Check if the folders contain matching WAV files.")

def main():
    parser = argparse.ArgumentParser(description="Calculate average SI-SDR for WAV files in specified folders.")
    parser.add_argument('--enhanced_folder','-e', type=str, required=True, help="Path to the folder containing enhanced WAV files")
    parser.add_argument('--clean_folder','-c', type=str, required=True, help="Path to the folder containing clean WAV files")
    args = parser.parse_args()
    
    calculate_average_si_sdr(args.clean_folder, args.enhanced_folder)

if __name__ == "__main__":
    main()