import json
import os
import base64
from typing import List, Dict

import requests
from PIL.Image import Image as PILImage
from datasets.features.image import encode_pil_image as pil_image_to_byte


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_pil_image(image: PILImage) -> str:
    return base64.b64encode(pil_image_to_byte(image)).decode('utf-8')


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
                        "text": "You are a helpful image analyst that specializes in localizing objects from satellite "
                                "and aerial images given a natural language instruction. "
                                "You always truthfully answer the user's question. If you are not sure about "
                                "something, don't answer false information."
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
        "max_tokens": 300,
        "temperature": 0.2
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if not response.ok:
        raise IOError(str(response.json()))

    return response.json()


if __name__ == "__main__":
    res = chat(
        encode_image("/home/danielz/PycharmProjects/vleo-bench/data/DIOR-RSVG/JPEGImages/14076.jpg"),
        "You are given an 800 x 800 satellite image. Identify the extent of the object in the description below in "
        "the format of [xmin, ymin, xmax, ymax], where the top-left coordinate is (x_min, y_min) and the bottom-right "
        "coordinate is (x_max, y_max). You should answer the extent without further explanation.\nDescription: A "
        "airplane on the upper left"
    )
    print(res)
