import argparse
import os
from scipy.io import wavfile
from pesq import pesq
from tqdm import tqdm
# from torchmetrics.audio import PerceptualEvaluationSpeechQuality
# import torch
'''
python calculate_PESQ.py --clean_folder wav/clean --enhanced_folder wav/noisy --mode wb
python calculate_PESQ.py --clean_folder ../VCTK-DEMAND/test/clean/ --enhanced_folder ../VCTK-DEMAND/test/enhanced/
'''

def calculate_pesq(clean_file, enhanced_file, mode):
    _, ref = wavfile.read(clean_file)
    _, deg = wavfile.read(enhanced_file)
    m=min(len(ref),len(deg))
    pesq_score = pesq(16000, ref[:m], deg[:m], mode)
    # wb_pesq = PerceptualEvaluationSpeechQuality(16000, mode)
    # pesq_score=wb_pesq(torch.from_numpy(deg[:m]), torch.from_numpy(ref[:m])).item()
    return pesq_score

def calculate_average_pesq(clean_folder, enhanced_folder, mode):
    pesq_list = []
    for clean_file in tqdm(os.listdir(clean_folder), desc='Average PESQ calculating'):
        if clean_file.endswith('.wav'):
            clean_path = os.path.join(clean_folder, clean_file)
            enhanced_path = os.path.join(enhanced_folder, clean_file)

            if not os.path.exists(enhanced_path):
                print(f"Enhanced file not found for: {clean_path}")
                continue

            pesq_score = calculate_pesq(clean_path, enhanced_path, mode)
            pesq_list.append(pesq_score)
            
            # print(f"File: {clean_file}, PESQ: {pesq_score}")

    if pesq_list:
        average_pesq = sum(pesq_list) / len(pesq_list)
        print("Average PESQ:", mode, average_pesq)
    else:
        print("No PESQ values calculated. Check if the folders contain matching WAV files.")

def main():
    parser = argparse.ArgumentParser(description="Calculate average PESQ scores for WAV files in specified folders.")
    parser.add_argument('--clean_folder', '-c', type=str, required=True, help="Path to the folder containing clean WAV files")
    parser.add_argument('--enhanced_folder', '-e', type=str, required=True, help="Path to the folder containing enhanced WAV files")
    parser.add_argument('--mode', '-m', type=str, choices=['wb', 'nb'], default='wb', help="PESQ mode: 'wb' for wideband or 'nb' for narrowband (default: 'wb')")
    args = parser.parse_args()
    
    calculate_average_pesq(args.clean_folder, args.enhanced_folder, args.mode)

if __name__ == "__main__":
    main()
