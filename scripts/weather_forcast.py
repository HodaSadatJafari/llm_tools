import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_current_weather(location: str) -> str:
    return f"The weather in {location} is sunny with a high of 25°C."


weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather for a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country (e.g. 'London, UK')",
                }
            },
            "required": ["location"],
        },
    },
}

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Use available functions when needed.",
    },
    {"role": "user", "content": "What's the weather in Tokyo?"},
]

# Make initial request
response = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, tools=[weather_tool], tool_choice="auto"
)

assistant_message = response.choices[0].message

# Handle function calls
if assistant_message.tool_calls:
    messages.append(assistant_message)

    for tool_call in assistant_message.tool_calls:
        function_args = json.loads(tool_call.function.arguments)
        function_result = get_current_weather(**function_args)

        messages.append(
            {"role": "tool", "tool_call_id": tool_call.id, "content": function_result}
        )

    final_response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )

    print("Assistant:", final_response.choices[0].message.content)
