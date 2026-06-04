"""
一键评测（并行版 v2）：对增强结果批量计算 PESQ、SI-SNR、STOI/eSTOI、DNSMOS

相比 batch_evaluate.py 的唯一区别：用 ProcessPoolExecutor 多进程并行计算，
默认同时开 8 个 worker（-j 8），可根据 CPU 核数调整。

用法:
    python batch_evaluate_parallel.py -c <clean_dir> -e <enhanced_dir>

示例:
    python batch_evaluate_parallel.py -c ../AlphaASR/examples/MCSE/beamformed_wav/clean \
                                      -e ../AlphaASR/examples/MCSE/beamformed_wav/DAS
    python batch_evaluate_parallel.py -c ../AlphaASR/examples/MCSE/beamformed_wav/clean \
                                      -e ../AlphaASR/examples/MCSE/beamformed_wav/DAS --extended
    python batch_evaluate_parallel.py -c clean -e enhanced -j 16       # 16 进程
    python batch_evaluate_parallel.py -c clean -e enhanced -j 4        # 4 进程
    python batch_evaluate_parallel.py -c clean -e enhanced --no-dnsmos # 跳过 DNSMOS
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi.stoi import stoi
from tqdm import tqdm

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

# ── SI-SNR (pure NumPy) ─────────────────────────────────────────────────────

EPS = 1e-8


def _remove_mean(x, axis=-1):
    return x - x.mean(axis=axis)


def _inner(a, b):
    return np.sum(a * b, axis=-1)


def _power_sum(x):
    return np.sum(x ** 2, axis=-1)


def si_snr(x, s):
    """Scale-invariant signal-to-noise ratio."""
    x = _remove_mean(x)
    s = _remove_mean(s)
    proj_x = (_inner(x, s) / _power_sum(s).clip(EPS)) * s
    n = x - proj_x
    return 10 * np.log10(_power_sum(proj_x).clip(EPS) / _power_sum(n).clip(EPS))


# ── Per-file worker (runs in subprocess) ────────────────────────────────────

def _compute_one(name, clean_path, enhanced_path, extended, do_dnsmos):
    """Compute PESQ, SI-SNR, STOI, (optionally DNSMOS) for one pair."""
    results = {"name": name, "pesq": None, "sisnr": None, "stoi": None,
               "OVRL": None, "SIG": None, "BAK": None, "P808_MOS": None}
    try:
        ref, rate = sf.read(clean_path)
        deg, _ = sf.read(enhanced_path)
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]

        # PESQ
        try:
            results["pesq"] = pesq(int(rate), ref, deg, "wb")
        except Exception:
            pass

        # SI-SNR
        try:
            results["sisnr"] = si_snr(deg, ref)
        except Exception:
            pass

        # STOI / eSTOI
        try:
            results["stoi"] = stoi(ref, deg, int(rate), extended=extended)
        except Exception:
            pass

        # DNSMOS (lazy-import inside subprocess to avoid blocking main)
        if do_dnsmos:
            try:
                import librosa
                import onnxruntime as ort
                DNSMOS_DIR = os.path.join(os.path.dirname(__file__), "DNSMOS")
                onnx_sess = ort.InferenceSession(
                    os.path.join(DNSMOS_DIR, "sig_bak_ovr.onnx"),
                    providers=["CPUExecutionProvider"],
                )
                p808_sess = ort.InferenceSession(
                    os.path.join(DNSMOS_DIR, "model_v8.onnx"),
                    providers=["CPUExecutionProvider"],
                )
                # segment the audio into 9.01s chunks
                fs = SAMPLING_RATE
                audio = librosa.resample(deg, int(rate), fs) if int(rate) != fs else deg.copy()
                len_samples = int(INPUT_LENGTH * fs)
                while len(audio) < len_samples:
                    audio = np.append(audio, audio)
                num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
                hop_len_samples = fs
                ovrl_list, sig_list, bak_list, p808_list = [], [], [], []
                for idx in range(num_hops):
                    seg = audio[idx * hop_len_samples: int((idx + INPUT_LENGTH) * hop_len_samples)]
                    if len(seg) < len_samples:
                        continue
                    inp = {"input_1": np.array(seg).astype("float32")[np.newaxis, :]}
                    raw_sig, raw_bak, raw_ovr = onnx_sess.run(None, inp)[0][0]
                    # polyfit calibration
                    p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
                    p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
                    p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
                    ovrl_list.append(p_ovr(raw_ovr))
                    sig_list.append(p_sig(raw_sig))
                    bak_list.append(p_bak(raw_bak))
                    # P808 MOS
                    mel = librosa.feature.melspectrogram(
                        y=seg[:-160], sr=fs, n_fft=321, hop_length=160, n_mels=120
                    )
                    mel_db = ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T.astype("float32")
                    p808_mos = p808_sess.run(None, {"input_1": mel_db[np.newaxis, :, :]})[0][0][0]
                    p808_list.append(p808_mos)
                if ovrl_list:
                    results["OVRL"] = float(np.mean(ovrl_list))
                    results["SIG"] = float(np.mean(sig_list))
                    results["BAK"] = float(np.mean(bak_list))
                    results["P808_MOS"] = float(np.mean(p808_list))
            except Exception:
                pass
    except Exception as e:
        print(f"  ERROR reading {name}: {e}")
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate enhancement metrics (parallel)")
    parser.add_argument("-c", "--clean_folder", required=True, help="Clean WAV folder")
    parser.add_argument("-e", "--enhanced_folder", required=True, help="Enhanced WAV folder")
    parser.add_argument("--extended", action="store_true", help="Use eSTOI instead of STOI")
    parser.add_argument("--no-dnsmos", action="store_true", help="Skip DNSMOS")
    parser.add_argument("-j", "--workers", type=int, default=8, help="Number of parallel processes (default 8)")
    args = parser.parse_args()

    clean_dir = args.clean_folder
    enhanced_dir = args.enhanced_folder

    clean_files = {os.path.basename(f): f for f in glob.glob(os.path.join(clean_dir, "*.wav"))}
    enhanced_files = {os.path.basename(f): f for f in glob.glob(os.path.join(enhanced_dir, "*.wav"))}
    common = sorted(set(clean_files) & set(enhanced_files))

    if not common:
        print("No matching WAV files found between clean and enhanced folders.")
        sys.exit(1)

    check_dnsmos = False
    if not args.no_dnsmos:
        dnsmos_model = os.path.join(os.path.dirname(__file__), "DNSMOS", "sig_bak_ovr.onnx")
        if not os.path.exists(dnsmos_model):
            print("  DNSMOS: model files not found in DNSMOS/, skipping.")
        else:
            check_dnsmos = True

    print(f"Found {len(common)} matching files")
    print(f"Clean:    {clean_dir}")
    print(f"Enhanced: {enhanced_dir}")
    print(f"Workers:  {args.workers}")
    print(f"DNSMOS:   {'yes' if check_dnsmos else 'no'}")
    print("─" * 50)

    # ── One parallel pass for ALL metrics ──
    all_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_compute_one, name, clean_files[name], enhanced_files[name],
                            args.extended, check_dnsmos): name
            for name in common
        }
        for f in tqdm(as_completed(futures), total=len(common), desc="Computing all metrics"):
            all_results.append(f.result())

    # ── Aggregate ──
    pesq_list = [r["pesq"] for r in all_results if r["pesq"] is not None]
    sisnr_list = [r["sisnr"] for r in all_results if r["sisnr"] is not None]
    stoi_list = [r["stoi"] for r in all_results if r["stoi"] is not None]
    ovrl_list = [r["OVRL"] for r in all_results if r["OVRL"] is not None]
    sig_list = [r["SIG"] for r in all_results if r["SIG"] is not None]
    bak_list = [r["BAK"] for r in all_results if r["BAK"] is not None]
    p808_list = [r["P808_MOS"] for r in all_results if r["P808_MOS"] is not None]

    print("\n" + "=" * 50)
    print("          Metric Summary")
    print("=" * 50)
    parts = []
    if pesq_list:
        parts.append(f"PESQ (wb): {np.mean(pesq_list):.4f}")
    if sisnr_list:
        parts.append(f"SI-SNR: {np.mean(sisnr_list):.4f}")
    if stoi_list:
        label = "eSTOI" if args.extended else "STOI"
        parts.append(f"{label}: {np.mean(stoi_list):.4f}")
    if parts:
        print("  " + ", ".join(parts))
    if ovrl_list:
        dnsmos_parts = [f"OVRL: {np.mean(ovrl_list):.4f}", f"SIG: {np.mean(sig_list):.4f}", f"BAK: {np.mean(bak_list):.4f}"]
        if p808_list:
            dnsmos_parts.append(f"P808_MOS: {np.mean(p808_list):.4f}")
        print(f"  DNSMOS {'/'.join(dnsmos_parts)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
