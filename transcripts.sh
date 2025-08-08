#!/bin/bash

INPUT_DIR="extracted_audio"
OUTPUT_DIR="transcripts"
MODEL="medium"

mkdir -p "$OUTPUT_DIR"

for audio in "$INPUT_DIR"/*.mp3; do
    [ -e "$audio" ] || continue

    echo "Transcribing: $audio"
    whisper "$audio" \
        --model "$MODEL" \
        --device cuda \
        --output_dir "$OUTPUT_DIR" \
        --output_format txt 
done

echo "✅ 转录完成：文本文件保存在 $OUTPUT_DIR/"
