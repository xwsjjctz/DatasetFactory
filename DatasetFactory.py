import os
import whisper
import opencc
import base64
import urllib
import requests
import json
from tqdm import tqdm
import librosa
import soundfile

import core
from slicer2 import Slicer
from common import *
from inference import inference_main
import resample
from inference_uvr5 import _audio_pre_

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
def cutwav(input):
    print("cutting...")
    core.working_threads(path=input, func=core.cutwav_core)
    print("success")

# 从媒体文件中抽取音频并转换成wav格式的音频文件
def data2wav(input):
    print("extracting audios...")
    core.working_threads(path=input, func=core.data2wav_core)
    print("success")

# 提取音频文件中的人声
def noise2vocal():
    print("extracting vocals...")
    filelist = file_path(WAVPATH)
    for file in filelist:
        input_path = os.path.join('.', WAVPATH, file)

        # inference_main(input_path)

        device = 'cuda'
        pre_fun = _audio_pre_(
            device=device,
            model_path = UVR5MODEL,
                            )
        audio_path = input_path
        save_path = input_path
        in_data , vo_data = pre_fun._path_audio_(audio_path , save_path)
        os.remove(input_path)
    print("success")

# 对人声进行切片
def wav2chunks():
    print("extracting chunks...")
    filelist = file_path(WAVPATH)
    for file in pbar(filelist):
        input_path = os.path.join('.', WAVPATH, file)
        audio, sr = librosa.load(input_path, sr=None, mono=False)
        slicer = Slicer(
            sr=sr,
            threshold=-40,
            min_length=5000,
            min_interval=300,
            hop_size=10,
            max_sil_kept=500
        )
        index_file = filelist.index(file)
        chunks = slicer.slice(audio)
        for j, chunk in enumerate(chunks):
            if len(chunk.shape) > 1:
                chunk = chunk.T
            if len(chunk)/sr > 3 and len(chunk)/sr < 13:
                soundfile.write(f"{input}{slash}{index_file+1}_{j+1}.wav", chunk, sr)
        os.remove(input_path)
    print("success")

# 识别人声并对应文件目录写入文本（利用whisper本地部署）
def whisper_speech2text():
    print("extracting texts...")
    filelist = file_path(WAVPATH)
    model = whisper.load_model(WHISPERMODEL)
    for file in pbar(filelist):
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
    wav_temp_list = file_path(WAVTEMPPATH)
    wav_list = file_path(WAVPATH)
    url = "https://vop.baidu.com/pro_api"
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    for wav_temp in pbar(wav_temp_list):
        index_wav_temp = wav_temp_list.index(wav_temp)
        wav = wav_list[index_wav_temp]
        try:
            wav_length = os.path.getsize(f"{WAVTEMPPATH}{slash}{wav_temp}")
            payload = json.dumps({
                "format": "wav",
                "rate": 16000,
                "channel": 1,
                "cuid": CUID,
                "token": get_access_token(),
                "dev_pid": 80001,
                "speech": get_file_content_as_base64(f"{WAVTEMPPATH}{slash}{wav_temp}", False),
                "len": wav_length
            })
            response = requests.request("POST", url, headers=headers, data=payload)
            response = json.loads(response.text)
            response = response["result"][0]
            if len(response) < 3:
                os.remove(f"{WAVTEMPPATH}{slash}{wav_temp}")
                os.remove(f"{WAVPATH}{slash}{wav}")
            else:
                FILELIST.append(f".{slash}dataset{slash}{SPEAKER}{slash}{wav}|{SPEAKER}|{LANGUAGE}|{response}{LANGUAGE}\n")
                os.remove(f"{WAVTEMPPATH}{slash}{wav_temp}")
        except KeyError:
            os.remove(f"{WAVTEMPPATH}{slash}{wav_temp}")
            os.remove(f"{WAVPATH}{slash}{wav}")
            continue
    read_File = open(FILELIST_FILE,'w',encoding='utf-8')
    read_File.writelines(FILELIST)
    read_File.close()
    print("success")

if __name__ == '__main__':
    # cutwav(DATASETPATH)
    # data2wav(DATASETPATH)
    noise2vocal()
    # wav2chunks()
    # # whisper_speech2text()
    # resample.wav_resample_multithreaded(16000)
    # baidu_speech2text()