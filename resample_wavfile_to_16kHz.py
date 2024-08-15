import librosa
import soundfile as sf
import os
import argparse

'''
python resample_wavfile_to_16kHz.py --input_file <path_to_your_wav_file>
'''

target_sr = 16000

parser = argparse.ArgumentParser(description="Resample a wav file to 16kHz.")
parser.add_argument("--input_file",'-i', type=str, help="The wav file to be resampled", required=True)

args = parser.parse_args()
input_file = args.input_file

file_dir, file_name = os.path.split(input_file)
file_base, file_ext = os.path.splitext(file_name)

if file_ext.lower() != ".wav":
    raise ValueError("Input file must be a .wav file")

output_file = os.path.join(file_dir, f"{file_base}_16k.wav")

wav_16k, _ = librosa.load(input_file, sr=target_sr)
sf.write(output_file, wav_16k, target_sr)
print(f"Resampled file saved to: {output_file}")
