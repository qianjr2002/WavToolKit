import argparse
from sigmos.sigmos import SigMOS
import librosa

'''
python evaluate_wavfile_sigmos.py -a ../dataset/clean_testset_wav/p232_041.wav
'''

def main():
    parser = argparse.ArgumentParser(description='Run the SigMOS estimator.')
    parser.add_argument('--audio_file', '-a', type=str, help='Path to the audio file to be processed')
    parser.add_argument('--model_dir', type=str, default='./sigmos', help='Directory where the model is stored')
    parser.add_argument('--sr', type=int, default=48000, help='Sampling rate to resample the audio to (default: 48000)')

    args = parser.parse_args()

    sigmos_estimator = SigMOS(model_dir=args.model_dir)

    audio, sr = librosa.load(args.audio_file, sr=args.sr)

    result = sigmos_estimator.run(audio, sr)
    print(result)


if __name__ == '__main__':
    main()