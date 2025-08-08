# VR-Ads
MAR2 competition track3

## 安装依赖
pip install -r requirements.txt

sudo apt update

sudo apt install ffmpeg

## 数据处理
### 1.提取音频
sh ./extracted_audio.sh
### 2.转录文本
sh ./transcripts.sh
### 3.视频分割
python video_segmentation.py
### 4.片段描述
python segment_description.py

## 推理
python inference.py
