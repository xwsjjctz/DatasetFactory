import multiprocessing
import concurrent.futures
import time
from common import file_path, DATASETPATH
from tqdm import trange
import os
import subprocess
import random
import math

cpu_threads = multiprocessing.cpu_count()
dataset = file_path(DATASETPATH)
num_threads = cpu_threads if len(dataset) > cpu_threads else len(dataset)

# 根据线程数来随机均分列表
def random_split(lst):  
    n = len(lst)
    m = math.ceil(n / num_threads)
    random.shuffle(lst)
    lst = [lst[i:i+m] for i in range(0, n, m)]
    return lst

dataset = random_split(dataset)
# print(dataset)

def cutwav(input):
    for i in trange(len(input)):
        file = os.path.join('.', DATASETPATH, input[i])
        file_format = os.path.splitext(file)[1]
        command = f'''ffprobe -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{file}" -v quiet'''
        duration = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
        duration = int(float(duration.stdout.strip()))
        if duration >= 1800:
            for i in range(0, duration, 1800):
                os.system(f'''ffmpeg -i "{file}" -ss {i} -t 1800 -c copy "{file}_{int(i/1800)}{file_format}" -v quiet''')
        os.remove(file)

if __name__ == "__main__":
    start = time.time()
    print("cutting...")
    with concurrent.futures.ThreadPoolExecutor(num_threads) as executor: 
        for i in dataset:
            # print(i)
            executor.submit(cutwav, i)
    print("success")
    end = time.time()
    spendtime = end - start
    print(spendtime)