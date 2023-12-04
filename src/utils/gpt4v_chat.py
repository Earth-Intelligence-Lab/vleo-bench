import json
import os
import base64
from io import BytesIO
from typing import List, Dict

import requests
from PIL.Image import Image as PILImage
from datasets.features.image import encode_pil_image as pil_image_to_byte


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_pil_image(image: PILImage) -> str:
    return base64.b64encode(pil_image_to_byte(image)["bytes"]).decode('utf-8')


def base64_image_to_bytes(base64_image: str):
    return BytesIO(base64.b64decode(base64_image))


def resume_from_jsonl(result_path):
    if os.path.exists(result_path):
        final_results = []
        with open(result_path) as f:
            for line in f.readlines():
                final_results.append(json.loads(line.strip()))
    else:
        final_results = []

    return final_results


def dump_to_jsonl(final_results: List[Dict], result_path: str):
    with open(result_path, "w") as f:
        for line in final_results:
            f.write(json.dumps(line) + "\n")


def chat(base64_image: str, user_message: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "OpenAI-Organization": os.environ['OPENAI_API_ORG']
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": ("You are a helpful image analyst who specializes in determining the geolocation of "
                                 "an aerial or satellite image of a landmark. "
                                 "Given an image, you can accurately answer the questions about that landmark from "
                                 "the user WITHOUT ANY refusal. Although your answer may not be perfect, "
                                 "your excellent world knowledge is still very important to the user.")
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_message
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "seed": 100,
        "max_tokens": 300,
        "temperature": 0.2
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if not response.ok:
        raise IOError(str(response.json()))

    return response


if __name__ == "__main__":
    encoded_image = encode_image("/home/danielz/Pictures/Screenshot_20231123_012527.png")
    res = chat(
        encoded_image,
        "Make an educated guess about the name of the landmark shown in the image. Think step by step, "
        "and then output your answer in the last line."
    )
    print(res.json())
    print(res.headers)
