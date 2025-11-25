# https://pypi.org/project/wvmos/
'''
conda create -n wvmos python=3.9 -y
pip install numpy==1.24.4
pip install torch==1.11.0
pip install wvmos==1.0
'''

'''
# 计算单个文件的MOS分数
python evaluate_wvmos.py --file "wav/noisy/p232_001.wav"

# 计算目录下所有音频文件的平均MOS分数
python evaluate_wvmos.py --dir "wav/noisy"

# 计算目录下所有音频文件的MOS分数（不计算平均值）
python evaluate_wvmos.py --dir "wav/noisy" --no-mean

# 不使用GPU加速
python evaluate_wvmos.py --dir "wav/noisy" --no-cuda
'''

import argparse
import os
import torch
import pytorch_lightning
torch.serialization.add_safe_globals([
    pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint
])

from wvmos import get_wvmos

def calculate_mos(file_path=None, dir_path=None, mean=True, cuda=True):
    """
    计算音频文件的MOS分数
    
    Args:
        file_path: 单个音频文件路径
        dir_path: 音频目录路径
        mean: 对于目录是否计算平均分数
        cuda: 是否使用GPU加速
    """
    model = get_wvmos(cuda=cuda)
    
    if file_path:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        mos = model.calculate_one(file_path)
        print(f"WVMOS分数: {mos:.4f}")
        return mos
        
    elif dir_path:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"目录不存在: {dir_path}")
        
        mos = model.calculate_dir(dir_path, mean=mean)
        if mean:
            print(f"平均MOS分数: {mos:.4f}")
        else:
            print(f"各文件MOS分数: {mos}")
        return mos
        
    else:
        raise ValueError("必须指定 --file 或 --dir 参数")

def main():
    parser = argparse.ArgumentParser(description="计算音频MOS分数")
    
    # 互斥参数：文件或目录
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="单个音频文件路径")
    group.add_argument("--dir", type=str, help="音频目录路径")
    
    # 可选参数
    parser.add_argument("--no-mean", action="store_true", 
                       help="对于目录，不计算平均分数（返回所有文件的分数）")
    parser.add_argument("--no-cuda", action="store_true", 
                       help="不使用GPU加速")
    
    args = parser.parse_args()
    
    try:
        calculate_mos(
            file_path=args.file,
            dir_path=args.dir,
            mean=not args.no_mean,
            cuda=not args.no_cuda
        )
    except Exception as e:
        print(f"错误: {e}")
        exit(1)

if __name__ == "__main__":
    main()