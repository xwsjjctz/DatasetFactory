import os
import whisper
import opencc
import base64
import urllib
import requests
import json
import multiprocessing
from tqdm import trange, tqdm
import librosa
import soundfile
from slicer2 import Slicer
from common import slash, file_path
from resample import wav_resample_multithreaded
from inference import inference_main

# 说话人名称
SPEAKER = "ishtar"
# 说话人ID
SPEAKER_ID = "2"
# 语言
LANGUAGE = "[ZH]"
# 百度云服务相关信息
API_KEY = ""
SECRET_KEY = ""
CUID = ""

WHISPERMODEL = os.path.join('.', 'models', 'large-v2.pt')           # whisper模型路径
FILELIST_FILE = os.path.join('.', 'filelist', 'temp.txt')           # 数据集输出文件
DATASETPATH = "dataset_raw"                                         # 输入文件夹
WAVPATH = "wav"                                                     # 输出文件夹
WAVTEMPPATH = "wav_temp"                                            # 输出文件夹中wav转换后的临时文件夹地址
FILELIST = []                                                       # speech2text函数输出缓存

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

# 从媒体文件中抽取音频并转换成wav格式的音频文件
def video2wav(input, output):
    print("extracting audios...")
    dataset = file_path(input)
    for i in trange(len(dataset)):
        input_path = os.path.join('.', input, dataset[i])
        output_path = os.path.join('.', output, dataset[i])
        if os.path.splitext(input_path)[1] == ".wav":
            os.rename(input_path, output_path)
        else:
            os.system(f'''ffmpeg -i "{input_path}" -vn -sn -c:a copy -y -map 0:a:0 "{output_path}.aac" -v quiet''')
            # os.system(f'''ffmpeg "{output_path}.wav" -i "{output_path}.aac" -v quiet''')
            # os.remove(f"{output_path}.aac")
    print("success")

# 提取音频文件中的人声
def noise2vocal(input):
    print("extracting vocals...")
    wav = file_path(WAVPATH)
    with tqdm(total=len(wav), desc='Total Progress') as pbar_total:
        for i in range(len(wav)):
            input_path = os.path.join('.', input, wav[i])
            inference_main(input_path)
            os.remove(input_path)
            pbar_total.update(1)
    print("success")

# 对人声进行切片
def wav2chunks(input):
    print("extracting chunks...")
    wav = file_path(WAVPATH)
    for i in trange(len(wav)):
        input_path = os.path.join('.', input, wav[i])
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
def whisper_speech2text(input):
    print("extracting texts...")
    wav = file_path(WAVPATH)
    model = whisper.load_model(WHISPERMODEL)
    for i in trange(len(wav)):
        result = model.transcribe(f"{input}{slash}{wav[i]}", language = "Chinese", fp16 = True)
        transcript = result["text"]
        converter = opencc.OpenCC('t2s')
        transcript = converter.convert(transcript)
        if len(transcript) < 3:
            os.remove(f"{input}{slash}{wav[i]}")
        else:
            FILELIST.append(f".{slash}dataset{slash}{SPEAKER}{slash}{wav[i]}|{SPEAKER_ID}|{LANGUAGE}{transcript}{LANGUAGE}\n")
    read_File = open(FILELIST_FILE,'w',encoding='utf-8')
    read_File.writelines(FILELIST)
    read_File.close()
    print("success")

# 识别人声并对应文件目录写入文本（调用百度语音识别的api）
def baidu_speech2text(input):
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
            wav_length = os.path.getsize(f"{input}{slash}{wav_temp[i]}")
            payload = json.dumps({
                "format": "wav",
                "rate": 16000,
                "channel": 1,
                "cuid": CUID,
                "token": get_access_token(),
                "dev_pid": 80001,
                "speech": get_file_content_as_base64(f"{input}{slash}{wav_temp[i]}", False),
                "len": wav_length
            })
            response = requests.request("POST", url, headers=headers, data=payload)
            response = json.loads(response.text)
            response = response["result"][0]
            if len(response) < 3:
                os.remove(f"{input}{slash}{wav_temp[i]}")
                os.remove(f"{WAVPATH}{slash}{wav[i]}")
            else:
                FILELIST.append(f".{slash}dataset{slash}{SPEAKER}{slash}{wav_temp[i]}|{SPEAKER_ID}|{LANGUAGE}{response}{LANGUAGE}\n")
                os.remove(f"{input}{slash}{wav_temp[i]}")
        except KeyError:
            os.remove(f"{input}{slash}{wav_temp[i]}")
            os.remove(f"{WAVPATH}{slash}{wav[i]}")
            continue
    read_File = open(FILELIST_FILE,'w',encoding='utf-8')
    read_File.writelines(FILELIST)
    read_File.close()
    print("success")

if __name__ == '__main__':
    res = 16000
    num_threads = multiprocessing.cpu_count()
    try:
        video2wav(DATASETPATH, WAVPATH)
        noise2vocal(WAVPATH)
        wav2chunks(WAVPATH)
        # whisper_speech2text(WAVPATH)
        wav_resample_multithreaded(WAVPATH, WAVTEMPPATH, res=res, num_threads=num_threads/2)
        baidu_speech2text(WAVTEMPPATH)
    except KeyboardInterrupt:
        print("stop execution")
        quit()
    finally:
        print("Program has finished")