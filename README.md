# DatasetFactory

该工具为文本生成语音技术提供数据集的处理，方便快速处理好数据集并开始训练。

安装依赖：
```
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
或者
```
conda env create -f conda_env.yml
```
推荐使用anaconda，或者对照conda_env.yml文件中的依赖进行安装。

DatasetFactory.py文件中有注释掉的whisper语音识别，速度慢，中文识别正确率较低。默认使用百度云的语音识别，需要去申请对应API。

提取人声部分使用uvr5，在models/文件夹下放入uvr5的权重文件即可。

所有个人修改的参数可以在common.py文件中找到。

将音视频文件放入dataset_raw/文件夹中。
使用以下命令运行。
```
python DatasetFactory.py
```
wav/文件夹下的文件为处理好的音频文件，filelist文件夹下的temp.txt文件为处理好的文本文件。

引用：
https://github.com/yang123qwe/vocal_separation_by_uvr5
https://github.com/openvpi/audio-slicer
