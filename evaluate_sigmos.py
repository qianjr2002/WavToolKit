import argparse
import os
from sigmos.sigmos import SigMOS
import librosa
from tqdm import tqdm

'''
python evaluate_sigmos.py --audio_dir ./wav/clean
'''

def main():
    parser = argparse.ArgumentParser(description='Run the SigMOS estimator on all audio files in a directory.')
    parser.add_argument('--audio_dir', type=str, required=True, help='Path to the directory containing audio files')
    parser.add_argument('--model_dir', type=str, default='./sigmos', help='Directory where the model is stored')
    parser.add_argument('--sr', type=int, default=48000, help='Sampling rate to resample the audio to (default: 48000)')

    args = parser.parse_args()

    sigmos_estimator = SigMOS(model_dir=args.model_dir)
    total_results = {'MOS_COL': 0, 'MOS_DISC': 0, 'MOS_LOUD': 0, 'MOS_NOISE': 0, 'MOS_REVERB': 0, 'MOS_SIG': 0, 'MOS_OVRL': 0}
    file_count = 0

    # Iterate over all files in the specified directory
    for filename in tqdm(os.listdir(args.audio_dir)):
        if filename.endswith('.wav'):  # Assuming audio files are in WAV format
            file_path = os.path.join(args.audio_dir, filename)
            # Load the audio file
            audio, sr = librosa.load(file_path, sr=args.sr)
            # Run the SigMOS estimator
            result = sigmos_estimator.run(audio, sr)
            # Accumulate the results
            for key in total_results.keys():
                total_results[key] += result[key]
            file_count += 1

    if file_count > 0:
        # Calculate the average results
        avg_results = {key: total_results[key] / file_count for key in total_results.keys()}
        print("Average SigMOS Results:")
        for key, value in avg_results.items():
            print(f"{key}: {value:.4f}")
    else:
        print("No audio files found in the specified directory.")

if __name__ == '__main__':
    main()