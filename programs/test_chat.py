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
    )

    print(f"Model device {model.device}")  # test .to('cuda')
    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    # Generate response
    response = ""
    try:
        response_ids = model.generate(
            **inputs,
            max_new_tokens=512,
        )
        generated_text = tokenizer.decode(
            response_ids[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        response = generated_text.strip()

        print(f"***** Response: {response}")
        # Yield the response incrementally
        # for i in range(1, len(response) + 1):
        #     yield response[:i]
        for i in range(len(response)):
            yield response[: i + 1]

    except Exception as e:
        yield f"An error occurred: {str(e)}"


model_name = "Qwen/Qwen3-1.7B"
# "Qwen/Qwen3-4B"
# "Qwen/Qwen3-1.7B"
# "Qwen/Qwen3-32B"
# "google/gemma-3n-E2B"
# "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Input model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

with gr.Blocks() as demo:
    chat_interface = gr.ChatInterface(
        fn=generate_response,
        type="messages",
    )


if __name__ == "__main__":
    demo.launch()
