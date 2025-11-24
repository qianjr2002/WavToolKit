#!/bin/bash

if [ $# -ne 2 ]; then
    echo "用法: $0 <clean_dir> <enh_dir>"
    exit 1
fi

clean_dir=$1
enh_dir=$2

if [ ! -d "$clean_dir" ] || [ ! -d "$enh_dir" ]; then
    echo "错误: 指定的目录不存在"
    exit 1
fi

python calculate_PESQ.py -c "$clean_dir" -e "$enh_dir" --mode wb
python calculate_PESQ.py -c "$clean_dir" -e "$enh_dir" --mode nb
python calculate_SI-SNR.py -c "$clean_dir" -e "$enh_dir"
python calculate_STOI.py -c "$clean_dir" -e "$enh_dir"
python calculate_STOI.py -c "$clean_dir" -e "$enh_dir" --extended
python evaluate_dnsmos.py -t "$enh_dir"

conda activate wvmos
python evaluate_wvmos.py --dir "$enh_dir"
conda deactivate

