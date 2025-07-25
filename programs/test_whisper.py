"""
!pip install -U openai-whisper
!sudo apt install ffmpeg

source: https://github.com/openai/whisper
"""

import whisper

model = whisper.load_model("turbo")
result = model.transcribe("whisper_inputs/Saad_01.mp3")

# Specify the output file name
output_file = "whisper_outputs/transcription.txt"

# Write the transcription result to the file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(result["text"])

print(f"Transcription saved to {output_file}")

# print(result["text"])
