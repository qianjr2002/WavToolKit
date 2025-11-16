import argparse
import json
from jiwer import wer

'''
python calculate_WER.py -c wav/clean/clean_text.json -e wav/noisy/noisy_text.json
'''

def compute_wer(clean_text_path, enh_text_path):
    """
    overall_wer: 所有句子平均的 WER
    item_wers: 每条句子的 WER 字典
    """
    # 读取 JSON
    with open(clean_text_path, "r", encoding="utf-8") as f:
        clean_dict = json.load(f)
    with open(enh_text_path, "r", encoding="utf-8") as f:
        noisy_dict = json.load(f)

    # 逐条计算 WER
    item_wers = {}
    wers = []

    for key in clean_dict:
        if key not in noisy_dict:
            continue

        ref = clean_dict[key]
        hyp = noisy_dict[key]

        w = wer(ref, hyp)
        item_wers[key] = w
        wers.append(w)

    # 平均 WER
    overall_wer = sum(wers) / len(wers) if wers else None
    return overall_wer, item_wers

def main(args):
    clean_text = args.clean_text
    enh_text = args.enh_text
    overall_wer, item_wers = compute_wer(clean_text, enh_text)
    print("Overall WER:", overall_wer)
    # print("Item WERs:", item_wers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--clean_text', default='wav/clean/clean_text.json')
    parser.add_argument('-e', '--enh_text', default='wav/noisy/noisy_text.json')
    args = parser.parse_args()
    main(args)
