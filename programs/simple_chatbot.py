"""
A simple chatbot
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient

# import torch

# Load environment variables in a file called .env
# Print the key prefixes to help with any debugging

load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")


# Initialize
openai = OpenAI()

# Load the model and tokenizer (you can change the model as needed)

# Close source
# MODEL_NAME = "gpt-4o-mini"

# Open source
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# MODEL_NAME = "microsoft/phi-2"
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # worked on my PC
# MODEL_NAME = (
#     "CohereLabs/aya-expanse-8b"  # https://huggingface.co/CohereLabs/aya-expanse-8b
# )

# MODEL_NAME = "google/gemma-3n-E2B"  # https://huggingface.co/google/gemma-3n-E2B

close_models = [
    "GPT 4o-mini",
]
open_models = [
    "Tiny Llama",
    "Mistral",
    "PHI",
    "AYA",
    "Hunyuan",
    "DeepSeek-R1",
    "Gemma 3n",
    "Qwen3 4B",
    "Qwen3-1.7B",
    "Qwen3-32B",
]
model_choices = open_models + close_models


system_message = "You are a helpful assistant"


def chat(message, history, selected_model):
    model_name = get_real_model_name(selected_model)
    print(f"Using this model: {model_name}")

    if selected_model in close_models:
        for value in chat_close_source(message, history, model_name):
            yield value
    else:
        for value in chat_open_source(message, history, model_name):
            yield value


def get_real_model_name(selected_model):
    match selected_model:
        case "GPT 4o-mini":
            return "gpt-4o-mini"
        case "Tiny Llama":
            return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        case "Mistral":
            return "mistralai/Mistral-7B-Instruct-v0.2"
        case "PHI":
            return "microsoft/phi-2"
        case "AYA":
            return "CohereLabs/aya-expanse-8b"
        case "Gemma 3n":
            return "google/gemma-3n-E2B"
        case "Hunyuan":
            return "tencent/Hunyuan-A13B-Instruct"
        case "DeepSeek-R1":
            return "deepseek-ai/DeepSeek-R1"
        case "Qwen3 4B":
            return "Qwen/Qwen3-4B"
        case "Qwen3-1.7B":
            return "Qwen/Qwen3-1.7B"
        case "Qwen3-32B":
            return "Qwen/Qwen3-32B"


def chat_close_source(message, history, model_name="gpt-4o-mini"):
    # MODEL_NAME = get_real_model_name(selected_model)
    # print(f"Using this model: {MODEL_NAME}")

    messages = (
        [{"role": "system", "content": system_message}]
        + history
        + [{"role": "user", "content": message}]
    )

    print("History is:")
    print(history)
    print("And messages is:")
    print(messages)

    stream = openai.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=True,
    )

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        yield response


def chat_open_source(message, history, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    # MODEL_NAME = get_real_model_name(selected_model)
    # print(f"Using this model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype=torch.float16,
        # device_map="auto", # "cpu"
        # torch_dtype=torch.float32,  # Use float32 for CPU
    )

    # Construct the full conversation context
    messages = (
        [{"role": "system", "content": system_message}]
        + history
        + [{"role": "user", "content": message}]
    )

    # Prepare the input for the model
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    # Tokenize the input
    inputs = tokenizer(
        input_text, padding="max_length", truncation=True, return_tensors="pt"
    ).to(model.device)

    # Generate response
    response = ""
    try:
        # Generate tokens
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=32768,  # Adjust as needed
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,  # Explicitly setting pad_token_id is also good practice
        )
        # Decode the generated tokens
        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        response = generated_text.strip()

        # Yield the response incrementally
        for i in range(1, len(response) + 1):
            yield response[:i]

    except Exception as e:
        yield f"An error occurred: {str(e)}"


def chat_open_source2(
    message, history, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
):
    # MODEL_NAME = get_real_model_name(selected_model)
    # print(f"Using this model: {MODEL_NAME}")

    # Construct the full conversation context
    messages = (
        [{"role": "system", "content": system_message}]
        + history
        + [{"role": "user", "content": message}]
    )

    client = InferenceClient(api_key=os.environ["HF_TOKEN"])

    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=True,
    )

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        yield response


# Create a Gradio interface with a dropdown for model selection
with gr.Blocks() as demo:
    # Dropdown for model selection
    model_dropdown = gr.Dropdown(
        choices=model_choices,
        label="Select Model",
        value=model_choices[0],  # Default to first model
    )

    # ChatInterface with model as an additional input
    chat_interface = gr.ChatInterface(
        fn=chat, type="messages", additional_inputs=[model_dropdown]
    )

# Launch the demo
demo.launch(inbrowser=True)
# , share=True
