# Load the dataset using the Hugging Face datasets library
import os
from dotenv import load_dotenv
from pathlib import Path
from datasets import load_dataset

# Load .env from parent directory
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Read the token
hf_token = os.getenv("HF_TOKEN")
print(hf_token)

# Load the full dataset
dataset = load_dataset("divarofficial/real_estate_ads")

# Print the first few examples
print(dataset["train"][:5])

# Get dataset statistics
print(f"Dataset size: {len(dataset['train'])} rows")
print(f"Features: {dataset['train'].features}")
