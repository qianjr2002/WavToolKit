import argparse
import os
import json
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from tqdm import tqdm

'''
python WhisperBatchASR.py -i wav/noisy -o wav/noisy/noisy_text.json
'''

class WhisperASR():
    def __init__(self, model_id="openai/whisper-large-v3-turbo", language="en"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # print(f"use device {self.device}")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=self.dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            dtype=self.dtype,
            device=self.device,
            return_timestamps=False,
            return_token_timestamps=False,
            language=language,
        )

    def transcribe(self, audio_path, **kwargs):
        return self.pipe(audio_path, **kwargs)


def main(args):
    input_dir = args.input_wav_dir
    output_json = args.output_test_json
    asr_model = WhisperASR()
    results = {}

    wav_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".wav")])

    print(f"Found {len(wav_files)} wav files.")

    for fname in tqdm(wav_files,desc='transcribing'):
        audio_path = os.path.join(input_dir, fname)
        key = os.path.splitext(fname)[0]   # 文件名去掉 .wav

        try:
            out = asr_model.transcribe(audio_path)
            text = out["text"]
        except Exception as e:
            print(f"Error processing {fname} : {e}")
            text = ""

        results[key] = text


    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Done! Saved to: {output_json}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_wav_dir', default='wav/clean')
    parser.add_argument('-o', '--output_test_json', default='wav/clean/clean_text.json')
    args = parser.parse_args()
    main(args)
