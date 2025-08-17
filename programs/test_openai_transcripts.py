import os
import requests
from dotenv import load_dotenv


load_dotenv(override=True)

url = "https://api.openai.com/v1/audio/transcriptions"
api_key = os.getenv("OPENAI_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}"
}

files = {
    "file": open("/home/ubuntu/llm_tools/programs/whisper_inputs/zeinab_denoise_boronfekani.wav", "rb")
}

data = {
    "model": "gpt-4o-transcribe"
}

response = requests.post(url, headers=headers, files=files, data=data)

print(response.json())

if not os.path.exists("openai_transcript_outputs"):
    os.makedirs("openai_transcript_outputs")

# save content to file
with open(f"openai_transcript_outputs/zeinab_denoise_boronfekani_transcript.txt", "w", encoding="utf-8") as f:
    f.write(response.json().get("text", "Transcript not found."))

