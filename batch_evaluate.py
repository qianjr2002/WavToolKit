"""
一键评测：对增强结果批量计算 PESQ、SI-SNR、STOI/eSTOI、DNSMOS

用法:
    python batch_evaluate.py -c <clean_dir> -e <enhanced_dir>

示例:
    python batch_evaluate.py -c ../AlphaASR/examples/MCSE/beamformed_wav/clean \
                             -e ../AlphaASR/examples/MCSE/beamformed_wav/DAS
    python batch_evaluate.py -c ../AlphaASR/examples/MCSE/beamformed_wav/clean \
                             -e ../AlphaASR/examples/MCSE/beamformed_wav/DAS --extended
"""
import argparse
import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch
from pesq import pesq
from pystoi.stoi import stoi
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from tqdm import tqdm

# ── Globals (created once, reused) ──────────────────────────────────────────
si_snr_metric = None

# ── DNSMOS ──────────────────────────────────────────────────────────────────
try:
    import onnxruntime as ort
    import librosa
    import pandas as pd
    import concurrent.futures
    _HAS_DNSMOS = True
except ImportError:
    _HAS_DNSMOS = False

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


class DNSMOSScore:
    def __init__(self, model_dir="DNSMOS"):
        self.onnx_sess = ort.InferenceSession(
            os.path.join(model_dir, "sig_bak_ovr.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.p808_onnx_sess = ort.InferenceSession(
            os.path.join(model_dir, "model_v8.onnx"),
            providers=["CPUExecutionProvider"],
        )

    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    @staticmethod
    def get_polyfit_val(sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return p_sig(sig), p_bak(bak), p_ovr(ovr)

    def __call__(self, fpath, is_personalized_MOS=False):
        aud, input_fs = sf.read(fpath)
        fs = SAMPLING_RATE
        audio = librosa.resample(aud, input_fs, fs) if input_fs != fs else aud
        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_ovr, predicted_mos_sig, predicted_mos_bak = [], [], []

        for idx in range(num_hops):
            seg = audio[int(idx * hop_len_samples): int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(seg) < len_samples:
                continue
            oi = {"input_1": np.array(seg).astype("float32")[np.newaxis, :]}
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            sig, bak, ovr = self.get_polyfit_val(mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS)
            predicted_mos_sig.append(sig)
            predicted_mos_bak.append(bak)
            predicted_mos_ovr.append(ovr)

        return {
            "OVRL": float(np.mean(predicted_mos_ovr)),
            "SIG": float(np.mean(predicted_mos_sig)),
            "BAK": float(np.mean(predicted_mos_bak)),
        }


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_pesq(clean_path, enhanced_path, mode="wb"):
    ref, rate = sf.read(clean_path)
    deg, _ = sf.read(enhanced_path)
    return pesq(int(rate), ref, deg, mode)


def compute_sisnr(clean_path, enhanced_path, device):
    clean, _ = sf.read(clean_path)
    enhanced, _ = sf.read(enhanced_path)
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]
    return si_snr_metric(
        torch.tensor(enhanced, device=device),
        torch.tensor(clean, device=device),
    ).item()


def compute_stoi(clean_path, enhanced_path, extended=False):
    ref, rate = sf.read(clean_path)
    deg, _ = sf.read(enhanced_path)
    min_len = min(len(ref), len(deg))
    return stoi(ref[:min_len], deg[:min_len], int(rate), extended=extended)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate enhancement metrics")
    parser.add_argument("-c", "--clean_folder", required=True, help="Clean WAV folder")
    parser.add_argument("-e", "--enhanced_folder", required=True, help="Enhanced WAV folder")
    parser.add_argument("--extended", action="store_true", help="Use eSTOI instead of STOI")
    parser.add_argument("--no-dnsmos", action="store_true", help="Skip DNSMOS")
    args = parser.parse_args()

    clean_dir = args.clean_folder
    enhanced_dir = args.enhanced_folder

    # Collect matching wav files
    clean_files = {os.path.basename(f): f for f in glob.glob(os.path.join(clean_dir, "*.wav"))}
    enhanced_files = {os.path.basename(f): f for f in glob.glob(os.path.join(enhanced_dir, "*.wav"))}
    common = sorted(set(clean_files) & set(enhanced_files))

    if not common:
        print("No matching WAV files found between clean and enhanced folders.")
        sys.exit(1)

    print(f"Found {len(common)} matching files")
    print(f"Clean:    {clean_dir}")
    print(f"Enhanced: {enhanced_dir}")
    print("─" * 50)

    # Set up GPU if available
    global si_snr_metric
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    si_snr_metric = ScaleInvariantSignalNoiseRatio().to(device)
    print(f"Using device: {device}\n")

    # Compute pair-wise metrics
    pesq_list, sisnr_list, stoi_list = [], [], []
    for name in tqdm(common, desc="Calculating PESQ / SI-SNR / STOI"):
        c_path = clean_files[name]
        e_path = enhanced_files[name]
        try:
            pesq_list.append(compute_pesq(c_path, e_path))
        except Exception as ex:
            print(f"  PESQ failed for {name}: {ex}")
        try:
            sisnr_list.append(compute_sisnr(c_path, e_path, device))
        except Exception as ex:
            print(f"  SI-SNR failed for {name}: {ex}")
        try:
            stoi_list.append(compute_stoi(c_path, e_path, args.extended))
        except Exception as ex:
            print(f"  STOI failed for {name}: {ex}")

    print("\n" + "=" * 50)
    print("          Metric Summary")
    print("=" * 50)
    parts = []
    if pesq_list:
        parts.append(f"PESQ{' (wb)' if True else ''}: {np.mean(pesq_list):.4f}")
    if sisnr_list:
        parts.append(f"SI-SNR: {np.mean(sisnr_list):.4f}")
    if stoi_list:
        label = "eSTOI" if args.extended else "STOI"
        parts.append(f"{label}: {np.mean(stoi_list):.4f}")
    if parts:
        print("  " + ", ".join(parts))

    # DNSMOS
    if not args.no_dnsmos and _HAS_DNSMOS:
        if not os.path.exists(os.path.join("DNSMOS", "sig_bak_ovr.onnx")):
            print("\n  DNSMOS: model files not found in DNSMOS/, skipping.")
        else:
            print("\n  DNSMOS (on enhanced folder) ...")
            try:
                scorer = DNSMOSScore()
                wavs = glob.glob(os.path.join(enhanced_dir, "*.wav"))
                ovrl_list, sig_list, bak_list = [], [], []
                for w in tqdm(wavs, desc="DNSMOS"):
                    result = scorer(w)
                    ovrl_list.append(result["OVRL"])
                    sig_list.append(result["SIG"])
                    bak_list.append(result["BAK"])
                print(f"  DNSMOS OVRL: {np.mean(ovrl_list):.4f}, SIG: {np.mean(sig_list):.4f}, BAK: {np.mean(bak_list):.4f}")
            except Exception as ex:
                print(f"  DNSMOS failed: {ex}")
    elif not _HAS_DNSMOS:
        print("\n  DNSMOS: onnxruntime/librosa/pandas not installed, skipping.")
    print("=" * 50)


if __name__ == "__main__":
    main()
