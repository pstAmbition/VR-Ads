"""
Video segmentation using PySceneDetect.
This script scans a directory of videos, detects scenes using content threshold,
splits the video into scenes, and saves metadata for each scene.
"""

from scenedetect import detect, ContentDetector, split_video_ffmpeg
import json
import os
from tqdm import tqdm

# ==== 配置部分 ====
video_dir = './adsqa_video_collection'
output_dir = './segment_videos'
# =================

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def video_segmentation(video):
    file_name = os.path.splitext(video)[0]
    video_path = os.path.join(video_dir, video)
    dst_dir = os.path.join(output_dir, file_name)
    ensure_dir(dst_dir)

    # 检测镜头
    scene_list = detect(video_path, ContentDetector(threshold=30.0))
    if not scene_list:
        print(f"No scenes detected in: {video}")

    # 分割视频
    split_video_ffmpeg(video_path, scene_list, output_file_template=os.path.join(dst_dir, 'scene_$SCENE_NUMBER.mp4'))

    # 记录每个镜头的信息
    scene_json = []
    for idx, (start, end) in enumerate(scene_list, 1):
        f_start = start.get_frames()
        f_end = end.get_frames()
        scene_json.append({
            "scene_id": idx,
            "start": str(start),
            "end": str(end),
            "start_frame": f_start,
            "end_frame": f_end,
            "duration_sec": (f_end - f_start) / 25.0
        })

    with open(os.path.join(dst_dir, file_name + ".json"), "w") as f:
        json.dump(scene_json, f, indent=2)

def main():
    ensure_dir(output_dir)
    video_list = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
    for video in tqdm(video_list, desc="Processing videos"):
        print(f"Processing: {video}")
        try:
            video_segmentation(video)
        except Exception as e:
            print(f"Error processing {video}: {e}")

if __name__ == "__main__":
    main()
