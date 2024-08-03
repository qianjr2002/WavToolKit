# WavToolKit
音频处理小工具

## 文件列表

- [wav_info_show.py](./wav_info_show.py)
    - 详细显示wav文件的信息

- [resample_to_16kHz.py](./resample_to_16kHz.py) 
    - 将wav文件重新采样到16kHz的脚本

- [plot_wav.py](./plot_wav.py)
    - 画出单通道wav文件的时域波形图以及频域功率谱图

- [trim_silence.py](./trim_silence.py)
    - 去除wav文件中的静音段，默认阈值为30db

- [calculate_wavfile_SI-SNR.py](./calculate_wavfile_SI-SNR.py)
    - 计算一对wav文件的SI-SNR值

- [calculate_SI-SNR.py](./calculate_SI-SNR.py)
    - 遍历指定文件夹中的所有wav文件，计算每个文件的SI-SNR值以及平均SI-SNR值

- [calculate_wavfile_SI-SNR.py](./calculate_wavfile_PESQ.py)
    - 计算一对wav文件的PESQ值

- [calculate_SI-SNR.py](./calculate_PESQ.py)
    - 遍历指定文件夹中的所有wav文件，计算每个文件的PESQ值以及平均PESQ值

- [calculate_wavfile_STOI.py](./calculate_wavfile_STOI.py)
    - 计算一对wav文件的STOI值

- [](./calculate_STOI.py)
    - 遍历指定文件夹中的所有wav文件，计算每个文件的STOI值以及平均STOI值

- [dnsmos_local_no_csv.py](./dnsmos_local_no_csv.py)
    - 遍历指定文件夹中的所有wav文件，计算平均BAK, SIG, OVRL等值