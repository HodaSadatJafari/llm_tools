import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Initialize tokenizer separately to apply chat template
MODEL_NAME = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Just in case

# Define a batch of prompts (4 in total)
user_questions = [
    "What is Thailand's national food symbol?",
    "What is the capital of Norway?",
    "Who invented the telescope?",
    "What is the chemical symbol for gold?",
]

# Create structured chat messages
messages_batch = [
    [
        {
            "role": "system",
            "content": "You are a helpful assistant. Respond in two sentences.",
        },
        {"role": "user", "content": question},
    ]
    for question in user_questions
]

# Manually apply chat template to each prompt
rendered_prompts = [
    tokenizer.apply_chat_template(messages, tokenize=False)
    for messages in messages_batch
]

# Define generation parameters
sampling_params = SamplingParams(max_tokens=128)

# Initialize vLLM
llm = LLM(
    model=MODEL_NAME,
    max_model_len=2048,
)

# Run batch generation
outputs = llm.generate(rendered_prompts, sampling_params)
print(outputs)
