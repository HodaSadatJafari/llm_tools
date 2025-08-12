import os
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
from dotenv import load_dotenv

# Constants
load_dotenv()
YOUTUBE_URL = os.getenv("YOUTUBE_URL")
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o-mini")

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def extract_video_id(url):
    """
    Extract the video ID from a YouTube URL.

    Args:
        url (str): The YouTube URL.

    Returns:
        str: The extracted video ID.

    Raises:
        ValueError: If the URL is invalid.
    """
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL")


def fetch_transcript(video_id):
    """
    Fetch the transcript of a YouTube video.

    Args:
        video_id (str): The YouTube video ID.

    Returns:
        str: The concatenated transcript text.
    """
    transcript = YouTubeTranscriptApi().fetch(video_id)
    return " ".join(entry["text"] for entry in transcript)


def save_to_file(filename, content):
    """
    Save content to a file.

    Args:
        filename (str): The name of the file.
        content (str): The content to save.
    """

    # create outputs directory if it doesn't exist
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    # save content to file
    with open(f"outputs/{filename}", "w", encoding="utf-8") as f:
        f.write(content)


def summarize_text(client, text):
    """
    Summarize the given text using the OpenAI API.

    Args:
        client (OpenAI): The OpenAI client.
        text (str): The text to summarize.

    Returns:
        str: The summarized text.
    """
    prompt = (
        "Please summarize the content of this YouTube transcript in a clear, concise, "
        "and professional manner. Highlight the key points, main ideas, and any notable "
        "conclusions or recommendations discussed. Exclude unnecessary details, repetition, "
        "and filler content, while retaining the essential information. Provide the summary "
        "in paragraph form, and ensure it is structured logically for easy comprehension."
        "The summary should be less than 500 words or 5000 characters."
        f'"""{text}"""'
    )
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


def main():
    """
    Main function to extract video ID, fetch transcript, and generate a summary.
    """
    try:
        # Extract video ID
        video_id = extract_video_id(YOUTUBE_URL)
        print(f"Video ID: {video_id}")

        # Fetch transcript
        transcript_text = fetch_transcript(video_id)
        save_to_file(f"transcript_{video_id}.txt", transcript_text)

        # Summarize transcript
        summary = summarize_text(client, transcript_text)
        save_to_file(f"summary_{video_id}.txt", summary)

        print("Transcript and summary saved successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
