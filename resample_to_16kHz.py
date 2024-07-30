import librosa
import soundfile as sf
import os
import numpy as np
import argparse
from tqdm import tqdm

"""
python resample_to_16kHz.py --wav_folder <the wavs folder directory>
"""

# Set sample rate before and after resampling
original_sr = 48000
target_sr = 16000

# Set the directory of folder of wav files to be resampled and get the list of these wav files
parser = argparse.ArgumentParser()
parser.add_argument("--wav_folder", type=str, help="The wav folder in which the wav files are to be resampled")

args = parser.parse_args()
wav_folder = args.wav_folder

wav_list = os.listdir(wav_folder)

# Create the folder of resampled files
wav_folder_16k = wav_folder + '_16k'
os.makedirs(wav_folder_16k, exist_ok=True)

# Start resampling
for wav in tqdm(wav_list, desc="Resampling to 16kHz"):
    # READ + RESAMPLE by librosa
    wav_path = os.path.join(wav_folder, wav)
    wav_16k, _ = librosa.load(wav_path, sr=target_sr)

    # WRITE by soundfile
    output_path = os.path.join(wav_folder_16k, wav)
    sf.write(output_path, wav_16k, target_sr)

