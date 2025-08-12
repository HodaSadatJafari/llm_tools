import os
import time
import argparse
import openai
from openai import OpenAI
from typing import List
from dotenv import load_dotenv


load_dotenv(override=True)

client = OpenAI()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError(
        "OpenAI API key not found. Set the OPENAI_API_KEY environment variable."
    )

# ------- Helpers -------


def read_transcript(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_output(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def chunk_text(text: str, max_tokens: int = 3000) -> List[str]:
    """
    Naive chunking by characters to stay within token-like limits.
    Adjust heuristics as needed; for more accurate token counting, integrate tiktoken.
    """

    approx_chars = max_tokens
    chunks = []
    start = 0
    while start < len(text):
        end = start + approx_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
        # try to break at nearest newline or space to avoid mid-sentence cuts
        split_at = text.rfind("\n", start, end)
        if split_at == -1:
            split_at = text.rfind(" ", start, end)
        if split_at == -1 or split_at <= start:
            split_at = end  # forced
        chunks.append(text[start:split_at].strip())
        start = split_at
    print(f"Len chunks: {len(chunks)}")
    return chunks


def build_prompt(transcript_chunk: str) -> str:
    """
    Customize this prompt for your formatting requirements.
    """
    return f"""
این متن حاصل ترنسکریپت یک فایل صوتی به زبان فارسی درباره مباحث قرآنی است. این متن شامل سخنان گوینده و آیات قرآن می‌باشد.
لطفاً متن را بدون کم یا زیاد کردن هیچ کلمه‌ای، فقط از نظر قالب‌بندی و ویرایش، به شکل کتابی تنظیم کن.
مراحل انجام:
ساختار پاراگراف‌ها را منظم کن تا خوانایی بیشتر شود.
آیات قرآن را دقیق و صحیح مطابق رسم‌الخط رایج قرآن بنویس.
بعد از هر آیه، مرجع سوره و شماره آیه را در پرانتز بیاور. مثال: «الم» (بقره:۱)
هیچ بخشی از متن حذف یا اضافه نشود. فقط قالب و علائم نگارشی و سجاوندی اصلاح شود.
گفتار محاوره‌ای را به نوشتار رسمی تغییر بده و از نظر نگارشی مرتب شود.

متن خام:
\"\"\"
{transcript_chunk}
\"\"\"
متن ویرایش شده:
"""


def call_openai_with_retry(
    prompt: str, model: str = "gpt-4", max_retries: int = 5
) -> str:
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You format transcripts as instructed.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=4000,  # adjust if needed
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError as e:
            print(f"RateLimitError on attempt {attempt}, backing off {backoff}s...")
            time.sleep(backoff)
            backoff *= 2
        except openai.OpenAIError as e:
            # for other transient errors
            print(f"OpenAI error on attempt {attempt}: {e}. Backing off {backoff}s...")
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(
        f"Failed to get response from OpenAI after {max_retries} attempts."
    )


def process_transcript(input_path: str, output_path: str, model: str = "gpt-4"):
    raw = read_transcript(input_path)
    chunks = chunk_text(raw)
    formatted_pieces = []

    for idx, chunk in enumerate(chunks, start=1):
        print(f"Processing chunk {idx}/{len(chunks)} (approx {len(chunk)} chars)...")
        prompt = build_prompt(chunk)
        formatted = call_openai_with_retry(prompt, model=model)
        formatted_pieces.append(formatted)
        # optional: small delay between chunks to be polite / avoid burst limits
        time.sleep(0.5)

    # Combine pieces. You might want to dedupe overlapping speaker labels or merge intelligently.
    final_output = "\n\n".join(formatted_pieces)
    write_output(output_path, final_output)
    print(f"Formatted transcript written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Format a transcript using OpenAI API."
    )
    parser.add_argument("input", help="Path to input transcript text file.")
    parser.add_argument("output", help="Path to write formatted transcript.")
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI model to use (e.g., gpt-4, gpt-3.5-turbo).",
    )
    args = parser.parse_args()

    process_transcript(args.input, args.output, model=args.model)


if __name__ == "__main__":
    main()
