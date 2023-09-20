from distutils import filelist
import multiprocessing
import concurrent.futures
import time
from common import file_path, DATASETPATH
from tqdm import tqdm
import os
import subprocess
import random
import math

cpu_threads = multiprocessing.cpu_count()
filelist = file_path(DATASETPATH)
num_threads = cpu_threads if len(filelist) > cpu_threads else len(filelist)

# 根据线程数来随机均分列表
def random_split(lst):  
    n = len(lst)
    m = math.ceil(n / num_threads)
    random.shuffle(lst)
    lst = [lst[i:i+m] for i in range(0, n, m)]
    return lst

filelist = random_split(filelist)
# print(dataset)

def cutwav_core(input):
    for file in tqdm(input):
        file_format = os.path.splitext(file)[1]
        filepath = os.path.join('.', DATASETPATH, file)
        command = f'''ffprobe -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{filepath}" -v quiet'''
        duration = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
        duration = int(float(duration.stdout.strip()))
        if duration >= 1800:
            for i in range(0, duration, 1800):
                os.system(f'''ffmpeg -i "{filepath}" -ss {i} -t 1800 -c copy "{filepath}_{int(i/1800)}{file_format}" -v quiet''')
        os.remove(filepath)

if __name__ == "__main__":
    start = time.time()
    print("cutting...")
    with concurrent.futures.ThreadPoolExecutor(num_threads) as executor: 
        for file in filelist:
            # print(i)
            executor.submit(cutwav_core, file)
    print("success")
    end = time.time()
    spendtime = end - start
    print(spendtime)