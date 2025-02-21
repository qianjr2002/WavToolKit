import wave
import os
import argparse
import hashlib

'''
python wav_info_show.py --wav_file <path to the wavs file>
'''

def get_wav_info(wav_file):
    with wave.open(wav_file, 'rb') as wf:
        encoding = 'pcm_s16le'  # 常见的WAV编码格式
        format_type = wf.getsampwidth()
        number_of_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        file_size = os.path.getsize(wav_file)
        duration = wf.getnframes() / sample_rate
        total_samples = wf.getnframes()

        return {
            'encoding': encoding,
            'format': format_type,
            'number_of_channels': number_of_channels,
            'sample_rate': sample_rate,
            'file_size': file_size,
            'duration': duration,
            'total_samples': total_samples
        }

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Show WAV file information.")
    parser.add_argument('--wav_file', type=str, required=True, help="Path to the WAV file")
    args = parser.parse_args()

    wav_info = get_wav_info(args.wav_file)
    md5_value = calculate_md5(args.wav_file)
    
    print(f"Encoding: {wav_info['encoding']}")
    print(f"Format: pcm_s{wav_info['format']*8}le")
    print(f"Number of Channels: {wav_info['number_of_channels']}")
    print(f"Sample Rate: {wav_info['sample_rate']} Hz")
    print(f"File Size: {wav_info['file_size']} bytes")
    print(f"Duration: {wav_info['duration']} seconds")
    print(f"Total Samples: {wav_info['total_samples']}")
    print(f"MD5: {md5_value}")

if __name__ == "__main__":
    main()