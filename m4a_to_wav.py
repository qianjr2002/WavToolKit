import argparse
from pydub import AudioSegment

'''
sudo apt update
sudo apt install ffmpeg
'''

'''
python m4a_to_wav.py wav/20240814_201408.m4a
'''

def convert_m4a_to_wav(input_file):
    audio = AudioSegment.from_file(input_file, format="m4a")
    output_file = input_file.rsplit(".", 1)[0] + ".wav"
    audio.export(output_file, format="wav")
    print(f"Converted {input_file} to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert M4A files to WAV format.")
    parser.add_argument("input_file", type=str, help="Path to the input M4A file.")
    args = parser.parse_args()
    
    convert_m4a_to_wav(args.input_file)

if __name__ == "__main__":
    main()
