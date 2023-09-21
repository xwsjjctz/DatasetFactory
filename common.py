import os
from natsort import natsorted, ns
from tqdm import tqdm

slash = os.path.sep                                                 # 系统文件路径分隔符
CUDALIST = 0                                                        # 选择gpu
WHISPERMODEL = os.path.join('.', 'models', 'large-v2.pt')           # whisper模型路径
FILELIST_FILE = os.path.join('.', 'filelist', 'temp.txt')           # 数据集输出文件
DATASETPATH = "dataset_raw"                                         # 输入文件夹
WAVPATH = "wav"                                                     # 输出文件夹
WAVTEMPPATH = "wav_temp"                                            # 输出文件夹中wav转换后的临时文件夹地址
# 说话人名称
SPEAKER = "azi"
# 语言
LANGUAGE = "[ZH]"
# 百度云服务相关信息
API_KEY = "NrV6ZYeG213XytOqkR21c6DB"
SECRET_KEY = "fzOmKHGEQC4lQGXn7jEOxBipGvvv2GgX"
CUID = "32857573"
FILELIST = []                                                       # speech2text函数输出缓存

# 返回目录所有文件的列表
def file_path(filepath):
    for path in os.walk(filepath):
        path = natsorted(path[2], alg=ns.PATH)
        return path
    
# 进度条
def pbar(input):
    return tqdm(input)