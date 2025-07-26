from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

system_message = "You are a helpful assistant"


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

    inputs = tokenizer(text, return_tensors="pt")
    response_ids = model.generate(**inputs, max_new_tokens=32768)[0][
        len(inputs.input_ids[0]) :
    ].tolist()
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    # Yield the response incrementally
    for i in range(1, len(response) + 1):
        yield response[:i]


model_name = "Qwen/Qwen3-4B"
# "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Input model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

with gr.Blocks() as demo:
    chat_interface = gr.ChatInterface(
        fn=generate_response,
        type="messages",
    )

# Launch the demo
demo.launch(inbrowser=True)
