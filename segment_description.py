from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import json
import subprocess
from glob import glob
from tqdm import tqdm

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")

def build_segment_visual_prompt(video_path, prompt_text, fps=1.0, max_pixels=360*420):
    system_prompt = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are a video analyst specialized in detailed multimodal observation. "
                    "Your job is to describe short video segments precisely. "
                    "Focus only on what can be directly seen or heard in each segment. "
                    "Do not infer or summarize beyond the segment content. "
                    "Be accurate, objective, and concise."
                )
            }
        ]
    }
    
    user_prompt = {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "fps": fps, "max_pixels": max_pixels},
            {"type": "text", "text": prompt_text}
        ]
    }

    return [system_prompt, user_prompt]


def get_answer(messages, max_tokens = 128):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]

def process_video(video_path, segment_dir):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    segment_dst = os.path.join(segment_dir, f"{video_id}_segment_descriptions.json")

    if os.path.exists(segment_dst):
        return 0


    segment_files = sorted([f for f in os.listdir(segment_dir) if f.endswith('.mp4')])

    print(segment_files)
    segment_descriptions = []

    print("🧠 正在进行 segment 描述生成...")
    for seg in segment_files:
        seg_path = os.path.join(segment_dir, seg)
        prompt = "Please describe the main characters, objects, and actions in this short video segment in no more than 20 words."
        messages = build_segment_visual_prompt(seg_path, prompt)
        try:
            desc = get_answer(messages, max_tokens=128)
            segment_descriptions.append({"file": os.path.basename(seg_path), "description": desc.strip()})
        except Exception as e:
            print(f"⚠️ Segment failed: {seg_path}: {e}")
            segment_descriptions.append({"file": os.path.basename(seg_path), "description": "ERROR"})

    print(segment_descriptions)
    print("💾 正在保存描述结果...")
    with open(segment_dst, "w") as f:
        json.dump(segment_descriptions, f, indent=2, ensure_ascii=False)

    print("✅ 处理完成")

# 示例调用
if __name__ == "__main__":
    input_video_dir = "./adsqa_video_collection"

    for video in tqdm(os.listdir(input_video_dir)):
        input_video_path = os.path.join(input_video_dir, video)
        video_id = os.path.splitext(os.path.basename(video))[0]

        segment_dir = "./segment_videos/" + video_id

        process_video(input_video_path, segment_dir)