#!/bin/bash

if [ $# -ne 4 ]; then
    echo "用法: $0 <clean_dir> <enh_dir> <clean_text> <enh_text>"
    exit 1
fi

clean_dir=$1
enh_dir=$2
clean_text=$3
enh_text=$4
conda activate whisper
python WhisperBatchASR.py -i "$clean_dir" -o "$clean_text"
python WhisperBatchASR.py -i "$enh_dir" -o "$enh_text"
python calculate_WER.py -c "$clean_text" -e "$enh_text"