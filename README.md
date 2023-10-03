# DatasetFactory

该工具为文本生成语音技术提供数据集的处理，方便快速处理好数据集并开始训练。

安装依赖：
```
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
或者
```
conda create env -f conda_env.yaml
```
推荐使用anaconda，或者对照conda_env.yaml文件中的依赖进行安装。

DatasetFactory.py文件中有注释掉的whisper语音识别，速度慢，中文识别正确率较低。默认使用百度云的语音识别，需要去申请对应API。

将音视频文件放入dataset_raw/文件夹中。
使用以下命令运行。
```
python DatasetFactory.py
```
wav/文件夹下的文件为处理好的音频文件，filelist文件夹下的temp.txt文件为处理好的文本文件。

引用：
https://github.com/yang123qwe/vocal_separation_by_uvr5
https://github.com/openvpi/audio-slicer
