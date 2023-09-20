import concurrent.futures
import multiprocessing
import librosa
from scipy.io import wavfile
from common import slash, file_path, WAVPATH, WAVTEMPPATH
import numpy as np
from tqdm import tqdm

num_threads = multiprocessing.cpu_count()

def resample_worker(res, wav_file):
    y, sr = librosa.load(f"{WAVPATH}{slash}{wav_file}", sr=None, mono=True)
    wav_files_res = librosa.resample(y, orig_sr=sr, target_sr=res)
    wavfile.write(f"{WAVTEMPPATH}{slash}{wav_file}", res, (wav_files_res * np.iinfo(np.int16).max).astype(np.int16))

def wav_resample_multithreaded(res, num_threads):
    print("resampling...")
    wav_files = file_path(WAVPATH)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(resample_worker, res, wav_file) for wav_file in wav_files]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass
        concurrent.futures.wait(futures)
    print("success")
