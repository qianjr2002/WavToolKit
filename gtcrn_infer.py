import os
import torch
import soundfile as sf
import argparse
from gtcrn import GTCRN

'''
python gtcrn_infer.py --ckpt_path gtcrn_checkpoints/model_trained_on_dns3.tar --input_wav  wav/noisy/p232_005.wav --output_wav wav/p232_005_enh.wav

python enhance_wav.py --ckpt_path gtcrn_checkpoints/model_trained_on_vctk.tar --input_wav wav/noisy/p232_005.wav
'''

def main():
    parser = argparse.ArgumentParser(description="Enhance noisy WAV file using GTCRN model")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument('--input_wav', '-i', type=str, required=True, help="Path to the noisy input WAV file")
    parser.add_argument('--output_wav', '-o', type=str, default=None, help="Path to save the enhanced output WAV file (optional)")
    args = parser.parse_args()

    input_wav_path = args.input_wav
    output_wav_path = args.output_wav if args.output_wav else os.path.join('wav/gtcrn_enh', os.path.basename(input_wav_path))

    ## load model
    device = torch.device("cpu")
    model = GTCRN().eval()
    ckpt_path = args.ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])

    ## load data
    mix, fs = sf.read(input_wav_path, dtype='float32')
    assert fs == 16000, "Sampling rate should be 16kHz"

    ## inference
    input = torch.stft(torch.from_numpy(mix), 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    with torch.no_grad():
        output = model(input[None])[0]
    enh = torch.istft(output, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)

    ## save enhanced wav
    sf.write(output_wav_path, enh.detach().cpu().numpy(), fs)
    print(f"Enhanced WAV saved to: {output_wav_path}")

if __name__ == "__main__":
    main()
