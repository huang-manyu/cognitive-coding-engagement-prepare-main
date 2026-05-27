import base64
import os
from pathlib import Path

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.pumpkinaigc.online/v1"),
    timeout=60,
)


def call_gpt(
    user_prompt: str,
    system_prompt: str = "你是个语言能力和逻辑理解能力很强的AI助手",
    model: str = "gpt-5",
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            timeout=300,
            reasoning_effort="high",
        )
        text = ""
        for chunk in stream:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                text += chunk.choices[0].delta.content
        return text
    except Exception as e:
        return str(e)


def call_gpt_vision(
    image_path: str | Path,
    user_prompt: str,
    system_prompt: str = "你是个语言能力和逻辑理解能力很强的AI助手",
    model: str = "gpt-5",
) -> str:
    image_path = Path(image_path)
    image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    suffix = image_path.suffix.lstrip(".").lower()
    media_type = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(suffix, "jpeg")

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{media_type};base64,{image_data}"},
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            timeout=300,
        )
        text = ""
        for chunk in stream:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                text += chunk.choices[0].delta.content
        return text
    except Exception as e:
        return str(e)
