import os
import argparse
import librosa
import soundfile as sf
from tqdm import tqdm

'''
python trim_silence.py -i wav/noisy --threshold_db 30
python trim_silence.py --input_dir wav/noisy --threshold_db 30
'''

def trim_silence(input_dir, threshold_db):
    output_dir = f"{input_dir}_trim_{threshold_db}db"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in tqdm(os.listdir(input_dir), desc="trimming"):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_dir, filename)
            y, fs = librosa.load(file_path, sr=None)
            yt, _ = librosa.effects.trim(y, top_db=threshold_db)
            output_path = os.path.join(output_dir, filename)
            sf.write(output_path, yt, fs)
            # print(f"Processed {filename} and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim silence from WAV files in a directory.")
    parser.add_argument('-i',"--input_dir", type=str, help="Path to the input directory containing WAV files.")
    parser.add_argument("--threshold_db", type=float, default=30, help="dB threshold for trimming silence (default: 30).")
    args = parser.parse_args()

    trim_silence(args.input_dir, args.threshold_db)
