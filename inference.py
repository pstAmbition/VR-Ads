from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import os

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")

import json
file_path = './adsqa_question_file.json'
with open(file_path, 'r', encoding='utf-8') as f:
    jsondata = json.load(f)
desfile_path = './video_audiotext.json'
with open(desfile_path, 'r', encoding='utf-8') as f:
    audiojsondata = json.load(f)  

def get_description_by_video_id(video_id, dataset):
    for item in dataset:
        if item.get("video_id") == video_id:
            return item.get("video_text")
    return None  # 如果找不到

def get_question_by_video_id(video_id, dataset):
    q_list = []
    for item in dataset:
        if item.get("video") == video_id:
            q_list.append(item)
    return q_list

def get_answer(messages, max_token = 128):
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
    generated_ids = model.generate(**inputs, max_new_tokens=max_token)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]

video_Q_dict = {}
for i in audiojsondata:
    id = i["video_id"]
    Q_list = get_question_by_video_id(id,jsondata)
    video_Q_dict[id] = Q_list

# 读入段落记忆文本
segment_dir = "./segment_videos"
def get_segment_memory_by_video_id(video_id):
    segment_file_path = f"{segment_dir}/{video_id}/{video_id}.json"
    segment_des_path = f"{segment_dir}/{video_id}/{video_id}_segment_descriptions.json"
    with open(segment_file_path, 'r', encoding='utf-8') as f:
        segment_file = json.load(f)
    with open(segment_des_path, 'r', encoding='utf-8') as f:
        segment_des = json.load(f)

    segment_memories = []
    for idx,(file_meta, des_meta) in enumerate(zip(segment_file, segment_des)):
        segment_memories.append(f"[Segment {idx+1}]: ({file_meta['start']},{file_meta['end']}),Description: {des_meta['description']}")
    return segment_memories



result = []
dst_path = './output_cot_72B.json'

if os.path.exists(dst_path):
    first = False
else:
    # 打开文件并清空原内容（如果你希望追加，请将 'w' 改为 'a' 并做相应调整）
    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write('[\n')  # 开始 JSON 数组

    first = True  # 标记是否是第一条，用于控制逗号

start = 0
for video_id, Q_list in tqdm(video_Q_dict.items()):
    print(video_id)
    filepath = f"./adsqa_video_collection/{video_id}.mp4"
    # 音频内容
    audio_text = get_description_by_video_id(video_id, audiojsondata)
    # print(f"audio text: {audio_text}")
    # 视频段落记忆
    segment_memories = get_segment_memory_by_video_id(video_id)
    if len(segment_memories) > 50:
        print(f"{video_id} too long")
        continue

    segment_memory_text = "\n".join(segment_memories)
    # 视频全局描述
    # 视觉描述
    visual_prompt = (
    "Please give an overall description of the key characters, actions, and emotions in the scene "
    "based on the visual and audio content of the video."
    f"textual information from the video's audio as follows: {audio_text}"
    )

    messages = [
        {
        "role": "system",
        "content": [{"type": "text", "text": (
            "You are an expert in advertising analysis and multimodal reasoning. "
            "Your task is to analyze video advertisements by combining visual elements, narrative structure, and any accompanying subtitles or text. "
        )}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": filepath,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": visual_prompt}
            ],
        }
    ]
    overall_description = get_answer(messages,1024)

    # 作为 memory 开始
    memory_chain = [f"[Overall Summary]: {overall_description}"]
    messages.append({"role": "assistant", "content": chr(10).join(memory_chain)})

    # cot引导问题回复
    for i, question_item in enumerate(Q_list):
        question = question_item['question']
        print(f"正在处理第{i+1}个问题，问题是：{question_item['question']}")
        # 构造引导性 CoT Prompt
        observation_prompt = f"""
            Based on the video and the following paragraph description, identify visual or auditory observations that are directly relevant to the question in 300 word.\n\n\n
            Question:
            {question}\n\n\n
            Paragraph Description:
            {segment_memory_text}
        """
        # print(observation_prompt)
        messages.append({"role": "user", "content": [{"type": "text", "text": observation_prompt}]})
        observation = get_answer(messages,512)
        print(f"observation:{observation}")

        reasoning_prompt = f"""
            Given the following question and observation, infer the underlying meaning or implication based on the ad's broader context in 300 word.\n\n\n
            Question:
            {question}\n\n\n
            Observation:
            {observation}
        """
        # print(reasoning_prompt)
        messages.append({"role": "user", "content": [{"type": "text", "text": reasoning_prompt}]})
        reasoning = get_answer(messages,256)
        print(f"reasoning:{reasoning}")

        ans_prompt = f"""
            You are generating a concise and well-reasoned answer based on the given question and reasoning.\n\n\n
            Instructions:\n
            - The answer must directly address the question.\n
            - The length must be between 25 and 30 words.\n\n\n
            Question:
            {question}\n\n\n
            Reasoning:
            {reasoning}
        """
        messages.append({"role": "user", "content": [{"type": "text", "text": ans_prompt}]})
        answer = get_answer(messages,256)
        print(f"第{i+1}个问题的答案是：{answer}\n")

        answer_item = {
            "question_id": question_item['question_id'],
            "question": question_item['question'],
            "answer": answer
        }
        # result.append(answer_item)
        print(answer_item)

        with open(dst_path, 'a', encoding='utf-8') as f:
            if not first:
                f.write(',\n')
            else:
                first = False
            json.dump(answer_item, f, ensure_ascii=False, indent=4)


# 结束 JSON 数组
with open(dst_path, 'a', encoding='utf-8') as f:
    f.write('\n]')

print(f"JSON 数据已写入 {dst_path}")