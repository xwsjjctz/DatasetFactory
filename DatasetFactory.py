from distutils import filelist
import os
import whisper
import opencc
import base64
import urllib
import requests
import json
import concurrent.futures
from tqdm import trange, tqdm
import librosa
import soundfile

from cutwav import dataset, num_threads, cutwav_core
from slicer2 import Slicer
from common import slash, file_path, WHISPERMODEL, FILELIST_FILE, \
    DATASETPATH, WAVPATH, WAVTEMPPATH, FILELIST, CUID, SECRET_KEY, API_KEY, LANGUAGE, SPEAKER
from inference import inference_main
import resample

res = 16000

# 音频转base64
def get_file_content_as_base64(path, urlencoded=False):
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content

# 百度云根据API_KEY和SECRET_KEY获取token
def get_access_token():
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

# 给长度过长的音频切片
def cutwav():
    with concurrent.futures.ThreadPoolExecutor(num_threads) as executor: 
        for i in dataset:
            # print(i)
            executor.submit(cutwav_core, i)

# 从媒体文件中抽取音频并转换成wav格式的音频文件
def data2wav():
    print("extracting audios...")
    filelist = file_path(DATASETPATH)
    for file in filelist:
        input_path = os.path.join('.', DATASETPATH, file)
        output_path = os.path.join('.', WAVPATH, file)
        if os.path.splitext(input_path)[1] == ".wav":
            os.rename(input_path, output_path)
        else:
            os.system(f'''ffmpeg -i "{input_path}" -vn -sn -c:a copy -y -map 0:a:0 "{output_path}.aac" -v quiet''')
            os.system(f'''ffmpeg "{output_path}.wav" -i "{output_path}.aac" -v quiet''')
            os.remove(f"{output_path}.aac")
    print("success")

# 提取音频文件中的人声
def noise2vocal():
    print("extracting vocals...")
    filelist = file_path(WAVPATH)
    with tqdm(total=filelist, desc='Total Progress') as pbar_total:
        for file in filelist:
            input_path = os.path.join('.', WAVPATH, file)
            inference_main(input_path)
            os.remove(input_path)
            pbar_total.update(1)
    print("success")

# 对人声进行切片
def wav2chunks():
    print("extracting chunks...")
    wav = file_path(WAVPATH)
    for i in trange(len(wav)):
        input_path = os.path.join('.', WAVPATH, wav[i])
        audio, sr = librosa.load(input_path, sr=None, mono=False)
        slicer = Slicer(
            sr=sr,
            threshold=-40,
            min_length=5000,
            min_interval=300,
            hop_size=10,
            max_sil_kept=500
        )
        chunks = slicer.slice(audio)
        for j, chunk in enumerate(chunks):
            if len(chunk.shape) > 1:
                chunk = chunk.T
            if len(chunk)/sr > 3 and len(chunk)/sr < 13:
                soundfile.write(f"{input}{slash}{i+1}_{j+1}.wav", chunk, sr)
        os.remove(input_path)
    print("success")

# 识别人声并对应文件目录写入文本（利用whisper本地部署）
def whisper_speech2text():
    print("extracting texts...")
    filelist = file_path(WAVPATH)
    model = whisper.load_model(WHISPERMODEL)
    for file in filelist:
        result = model.transcribe(f"{WAVPATH}{slash}{file}", language = "Chinese", fp16 = True)
        transcript = result["text"]
        converter = opencc.OpenCC('t2s')
        transcript = converter.convert(transcript)
        if len(transcript) < 3:
            os.remove(f"{WAVPATH}{slash}{file}")
        else:
            FILELIST.append(f".{slash}dataset{slash}{SPEAKER}{slash}{file}|{SPEAKER}|{LANGUAGE}|{transcript}{LANGUAGE}\n")
    read_File = open(FILELIST_FILE,'w',encoding='utf-8')
    read_File.writelines(FILELIST)
    read_File.close()
    print("success")

# 识别人声并对应文件目录写入文本（调用百度语音识别的api）
def baidu_speech2text():
    print("extracting texts...")
    wav_temp = file_path(WAVTEMPPATH)
    wav = file_path(WAVPATH)
    url = "https://vop.baidu.com/pro_api"
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    for i in trange(len(wav_temp)):
        try:
            wav_length = os.path.getsize(f"{WAVTEMPPATH}{slash}{wav_temp[i]}")
            payload = json.dumps({
                "format": "wav",
                "rate": 16000,
                "channel": 1,
                "cuid": CUID,
                "token": get_access_token(),
                "dev_pid": 80001,
                "speech": get_file_content_as_base64(f"{WAVTEMPPATH}{slash}{wav_temp[i]}", False),
                "len": wav_length
            })
            response = requests.request("POST", url, headers=headers, data=payload)
            response = json.loads(response.text)
            response = response["result"][0]
            if len(response) < 3:
                os.remove(f"{WAVTEMPPATH}{slash}{wav_temp[i]}")
                os.remove(f"{WAVPATH}{slash}{wav[i]}")
            else:
                FILELIST.append(f".{slash}dataset{slash}{SPEAKER}{slash}{wav[i]}|{SPEAKER}|{LANGUAGE}|{response}{LANGUAGE}\n")
                os.remove(f"{WAVTEMPPATH}{slash}{wav_temp[i]}")
        except KeyError:
            os.remove(f"{WAVTEMPPATH}{slash}{wav_temp[i]}")
            os.remove(f"{WAVPATH}{slash}{wav[i]}")
            continue
    read_File = open(FILELIST_FILE,'w',encoding='utf-8')
    read_File.writelines(FILELIST)
    read_File.close()
    print("success")

if __name__ == '__main__':
    # cutwav()
    data2wav()
    # noise2vocal()
    # wav2chunks()
    # whisper_speech2text()
    # resample.wav_resample_multithreaded(16000)
    # baidu_speech2text()