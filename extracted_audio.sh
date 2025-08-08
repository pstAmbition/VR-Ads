#!/bin/bash

mkdir -p extracted_audio

for video in adsqa_video_collection/*.mp4; do
    base_name="${video%.*}"
    ffmpeg -i "$video" -q:a 0 -map a "extracted_audio/${base_name}.mp3"
done
