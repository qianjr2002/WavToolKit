import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

'''
python plot_wav.py --wav_file <path to the wavs file>
'''

def plot_waveform_and_spectrum(wav_file):
    y, sr = librosa.load(wav_file, sr=None)

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    stft_result = librosa.stft(y)
    D = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
    
    plt.subplot(2, 1, 2)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.title('Power Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot waveform and power spectrum of a WAV file.")
    parser.add_argument('--wav_file', type=str, required=True, help="Path to the WAV file")
    args = parser.parse_args()
    
    plot_waveform_and_spectrum(args.wav_file)

if __name__ == "__main__":
    main()
