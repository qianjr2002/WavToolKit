import os
import torch
import soundfile as sf
from tqdm import tqdm
from gtcrn import GTCRN
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="Enhance audio files using GTCRN model")
parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint file")
parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing noisy wav files")
parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder to save enhanced wav files")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
model = GTCRN().eval().to(device)

ckpt = torch.load(args.ckpt_path, map_location=device)
model.load_state_dict(ckpt['model'])

input_folder = Path(args.input_folder)
output_folder = Path(args.output_folder)
output_folder.mkdir(parents=True, exist_ok=True)

wav_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]

def process_file(filename):
    try:
        input_path = input_folder / filename
        output_path = output_folder / filename

        mix, fs = sf.read(input_path, dtype='float32')
        assert fs == 16000, f"Sample rate mismatch: {fs} Hz (Expected: 16000 Hz)"

        input = torch.from_numpy(mix).to(device).contiguous()
        input = torch.stft(input, n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512, device=device), return_complex=True)
        input = torch.view_as_real(input).contiguous().unsqueeze(0)

        with torch.no_grad():
            output = model(input)[0]
            output = output.contiguous()
            output = torch.view_as_complex(output).contiguous()

            enh = torch.istft(output, n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512, device=device))

        sf.write(output_path, enh.cpu().numpy(), fs)
    except Exception as e:
        print(f"Error processing {filename}: {e}")

with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(process_file, wav_files), total=len(wav_files), desc="Processing files"))

print("All files processed.")
