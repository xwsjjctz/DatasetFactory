import multiprocessing
import concurrent.futures
from common import file_path, pbar, DATASETPATH, WAVPATH
import os
import subprocess
import random
import math

cpu_threads = multiprocessing.cpu_count()

def num_threads(input):
    return cpu_threads if len(file_path(input)) > cpu_threads else len(file_path(input))

# 根据线程数来随机均分列表
def random_split(input, lst):  
    n = len(lst)
    m = math.ceil(n / num_threads(input))
    random.shuffle(lst)
    lst = [lst[i:i+m] for i in range(0, n, m)]
    return lst

def filelist(input):
    return random_split(input, file_path(input))

# print(dataset)

def cutwav_core(input):
    for file in pbar(input):
        file_format = os.path.splitext(file)[1]
        filepath = os.path.join('.', DATASETPATH, file)
        command = f'''ffprobe -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{filepath}" -v quiet'''
        duration = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
        duration = int(float(duration.stdout.strip()))
        if duration >= 1800:
            for i in range(0, duration, 1800):
                os.system(f'''ffmpeg -i "{filepath}" -ss {i} -t 1800 -c copy "{filepath}_{int(i/1800)}{file_format}" -v quiet''')
            os.remove(filepath)

def data2wav_core(input):
    for file in pbar(input):
        input_path = os.path.join('.', DATASETPATH, file)
        output_path = os.path.join('.', WAVPATH, file)
        if os.path.splitext(input_path)[1] == ".wav":
            os.rename(input_path, output_path)
        else:
            os.system(f'''ffmpeg -i "{input_path}" -vn -sn -c:a copy -y -map 0:a:0 "{output_path}.aac" -v quiet''')
            os.system(f'''ffmpeg "{output_path}.wav" -i "{output_path}.aac" -v quiet''')
            os.remove(f"{output_path}.aac")

def working_threads(path, func):
    with concurrent.futures.ThreadPoolExecutor(num_threads(path)) as executor: 
        for file in filelist(path):
            # print(i)
            executor.submit(func, file)
