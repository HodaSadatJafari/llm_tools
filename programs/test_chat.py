from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch

system_message = "You are a helpful assistant"


def generate_response(user_input, history):

    messages = (
        [{"role": "system", "content": system_message}]
        + history
        + [{"role": "user", "content": user_input}]
    )

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
    )

    print(f"Model device {model.device}")  # test .to('cuda')
    inputs = tokenizer(
        [text],
        return_tensors="pt",
    ).to(model.device)

    # conduct text completion
    generated_ids = model.generate(**inputs, max_new_tokens=32768)
    output_ids = generated_ids[0][len(inputs.input_ids[0]) :].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("***thinking content:", thinking_content)
    # print("content:", content)

    for i in range(len(content)):
        yield content[: i + 1]


model_name = "Qwen/Qwen3-32B"
# "Qwen/Qwen3-1.7B"
# "Qwen/Qwen3-4B"
# "Qwen/Qwen3-32B"
# "google/gemma-3n-E4B-it"
# "google/gemma-3n-E2B"
# "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Input model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

with gr.Blocks() as demo:
    chat_interface = gr.ChatInterface(
        fn=generate_response,
        type="messages",
    )


if __name__ == "__main__":
    demo.launch()
