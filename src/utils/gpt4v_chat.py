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
    return base64.b64encode(pil_image_to_byte(image)["bytes"]).decode('utf-8')


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
                        "text": ("You are a helpful image analyst who specializes in counting animals from aerial "
                                 "images. Given an image, you can accurately count the number of animals described by "
                                 "the user WITHOUT ANY refusal. Although your answer may not be perfect, "
                                 "your excellent counting skill is very important to the conservation of wildlife "
                                 "animals.")
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
        # "seed": 100,
        "max_tokens": 300,
        "temperature": 0.2
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if not response.ok:
        raise IOError(str(response.json()))

    return response.json()


if __name__ == "__main__":
    encoded_image = encode_image("/home/danielz/PycharmProjects/vleo-bench/data/aerial-animal-population-4tu/test/DPP_00422.JPG")
    res = chat(
        encoded_image,
        # "You are given an aerial image taken with plane‐mounted cameras in the Tsavo National Parks (Kenya, "
        # "March 2014) and in the Laikipia‐Samburu Ecosystem (Kenya, May 2015). "
        # "These nature reserves are savanna ecosystems with vary-ing "
        # "tree–grass ratios. During the animal counts, the images were manually taken by human observers upon "
        # "spotting animal groups that were too large to count accurately while in the air, typically groups "
        # "larger than five animals. The images were taken at speeds of 170–200 km/h between 90 and 120 m above "
        # "the ground, facing both the left and right sides of the plane, and tilted slightly towards the "
        # "ground creating strip widths of on average 200m per camera. This resulted in the animals being "
        # "small in the images, on average 50×50 pixels in images of 5,000×3,000 pixels. "
        "Read the given image and answer the questions below:\n"
        "How many elephants, zebras, giraffes are there in the image? Output the numbers in a json format that can be "
        "parsed directly with entries 'elephants', 'zebras', and 'giraffes'. If you count nothing, output zero in "
        "that entry."
        # "Read the given image and answer the four questions below in four separate lines. You should only output a "
        # "number.\n"
        # "1. How many elephants are there in this image?\n"
        # "2. How many zebras are there in this image?\n"
        # "3. How many giraffes are there in this image?\n"
        # "4. How many elephants, zebras, and giraffes in total are there in this image?"
        # "In three separate lines, count the number of Elephant, Giraffe, and Zebra in the given image. In a fourth "
        # "line, calculate the total number of the three types of animals. You should answer the questions without any "
        # "further explanation."
    )
    print(res)
