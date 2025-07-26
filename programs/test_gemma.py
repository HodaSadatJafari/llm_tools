from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import requests
import torch


def generate_response(user_input, history):

    messages = (
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            }
        ]
        + history
        + [
            {
                "role": "user",
                "content": [
                    # {
                    #     "type": "image",
                    #     "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                    # },
                    {"type": "text", "text": user_input},
                ],
            },
        ]
    )

    inputs = model.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]

    decoded = model.decode(generation, skip_special_tokens=True)
    response = decoded.strip()

    # Yield the response incrementally
    for i in range(1, len(response) + 1):
        yield response[:i]


def generate_response(user_input, history):
    print(history)
    messages = (
        [{"role": "system", "content": system_message}]
        + history
        + [{"role": "user", "content": user_input}]
    )

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print(f"Model device {model.device}")  # test .to('cuda')
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    response_ids = model.generate(**inputs, max_new_tokens=32768)[0][
        len(inputs.input_ids[0]) :
    ].tolist()

    generated_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    response = generated_text.strip()

    # Yield the response incrementally
    for i in range(1, len(response) + 1):
        yield response[:i]


model_name = "google/gemma-3n-E2B"
print(f"Input model: {model_name}")
model = Gemma3nForConditionalGeneration.from_pretrained(
    model_name,
    device="cuda",
    torch_dtype=torch.bfloat16,
).eval()

model = AutoProcessor.from_pretrained(model_name)

with gr.Blocks() as demo:
    chat_interface = gr.ChatInterface(
        fn=generate_response,
        type="messages",
    )

# Launch the demo
demo.launch(inbrowser=True)
