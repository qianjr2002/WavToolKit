import os
import torch
import soundfile as sf
from tqdm import tqdm
from gtcrn import GTCRN
import argparse

'''
python gtcrn_batch_infer.py --ckpt_path gtcrn_checkpoints/model_trained_on_dns3.tar --input_folder ~/VCTK-DEMAND/test/noisy/ --output_folder ~/VCTK-DEMAND/test/enhanced2/
'''

# 解析命令行参数
parser = argparse.ArgumentParser(description="Enhance audio files using GTCRN model")
parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint file")
parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing noisy wav files")
parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder to save enhanced wav files")
args = parser.parse_args()

# 定义设备
device = torch.device("cpu")  # 如果有GPU并且希望使用，可以改为 "cuda"
model = GTCRN().eval()

# 加载预训练模型权重
ckpt_path = args.ckpt_path
ckpt = torch.load(ckpt_path, map_location=device)

model.load_state_dict(ckpt['model'])

# 定义输入和输出文件夹
input_folder = args.input_folder
output_folder = args.output_folder

# 创建输出文件夹如果不存在
os.makedirs(output_folder, exist_ok=True)

# 获取所有待处理的wav文件
wav_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]

# 遍历文件夹中的所有wav文件，并显示进度条
for filename in tqdm(wav_files, desc="Processing files"):
    # 构建文件路径
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    
    # 加载音频文件
    mix, fs = sf.read(input_path, dtype='float32')
    assert fs == 16000, "Sample rate of the input file must be 16000 Hz"
    
    # 进行短时傅里叶变换（STFT）
    input = torch.stft(torch.from_numpy(mix), n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512).pow(0.5), return_complex=False)
    
    # 使用模型进行增强
    with torch.no_grad():
        output = model(input[None])[0]
    
    # 进行逆短时傅里叶变换（ISTFT）
    enh = torch.istft(output, n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512).pow(0.5), return_complex=False)
    
    # 保存增强后的音频文件
    sf.write(output_path, enh.detach().cpu().numpy(), fs)

print("All files processed.")
