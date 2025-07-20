"""
!pip install -U openai-whisper
"""

import whisper

model = whisper.load_model("medium")
result = model.transcribe("/home/hoda/Desktop/llms_test/voices/farsi_woman_voice.mp3")
print(result["text"])
