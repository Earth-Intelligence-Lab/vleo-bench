import base64
import io
import os
import json
import random
import re
from glob import glob
import xml.etree.ElementTree as ET

import pandas as pd
from PIL import Image as PILImage

import numpy as np
from huggingface_hub.inference._text_generation import ValidationError
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm

from src.datasets.dataset import VLEODataset
from datasets import Dataset, Image, load_dataset

from src.utils.counting import parse_digit_response, calculate_counting_metrics, plot_scatter
from src.utils.gpt4v_chat import resume_from_jsonl, encode_image, dump_to_jsonl, encode_pil_image

SYS_PROMPT = ("You are an expert in satellite and aerial image analysis. You can see the earth by reading the object "
              "bounding boxes in satellite or aerial images, and answering the user's questions. Always answer as "
              "helpfully as possible. If a question does not make any sense or is not factually coherent, explain why "
              "instead of answering something not correct. If you don't know the answer to a question, please don't "
              "share false information.")

CAPTIONING_PROMPTS = (
    "You are given a {h}x{w} aerial image of a forest with {count} trees. Given a list of the bounding "
    "boxes (in xyxy format) of the trees in the scene, generate five captions, with each less than "
    "50 words to describe the image in general. You should try your best to reflect the spatial "
    "relationships between the trees. If you want to refer to a particular tree, use their bounding "
    "boxes like <bbox> [330 367 365 400] </bbox> instead of their IDs like \"Tree 0.\"\nTrees in "
    "the scene:\n{trees}\nYour answer should only be a JSON list that can be parsed directly. You "
    "should only include captions in your answers, not bounding boxes.")


class NeonTreeEvaluationDataset(VLEODataset):
    train_url = "https://zenodo.org/records/5914554/files/training.zip?download=1"
    train_labels_url = "https://zenodo.org/records/5914554/files/training.zip?download=1"
    test_url = "https://zenodo.org/records/5914554/files/evaluation.zip?download=1"

    dataset_base = "./data/NeonTreeEvaluation"
    annotation_path = "./data/NeonTreeEvaluation/annotations/"
    meta_path = "./data/NeonTreeEvaluation/RGB/metadata.jsonl"

    splits = ["train", "evaluation"]
    system_message = ("You are a helpful image analyst who specializes in counting trees from aerial images. "
                      "Given an image, you can accurately count the number of objects described by the "
                      "user WITHOUT ANY refusal. Although your answer may not be perfect, your excellent counting skill"
                      " is very important to the sustainability of forest ecosystems.")

    # "You always truthfully answer the user's question. If you are not sure about "
    # "something, don't answer false information."

    def __init__(self, credential_path: str, split: str = "evaluation"):
        super().__init__(credential_path)
        assert split in self.splits
        self.split = split

    def xml2jsonl(self, xml_file):
        # Parse the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Open the JSONL file for writing
        # Initialize objects dictionary
        objects = {"bbox": [], "categories": []}

        # Extract file name
        file_name = root.find('filename').text

        size_info, *_ = root.findall("size")

        # Iterate through all 'object' elements in the XML file
        for obj in root.findall('object'):
            # Extract bounding box coordinates and category
            bbox = obj.find('bndbox')
            # Assuming one bbox per object
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            objects["bbox"].append([xmin, ymin, xmax, ymax])

            category = obj.find('name').text
            objects["categories"].append(category)

        # Construct the final dictionary
        file_path = os.path.join(self.dataset_base, self.split, "RGB", file_name)
        output = {
            "image": file_path,
            "path": os.path.basename(file_name),
            "objects": objects,
            "count": len(objects["bbox"]),
            "height": int(size_info.find("height").text),
            "width": int(size_info.find("width").text)
        }

        return file_path, output

    def _load_metadata(self):
        annotation_files = glob(os.path.join(self.annotation_path, "*.xml"))
        print(annotation_files)

        img_metadata = []
        for xml_file in annotation_files:
            processed_name, metadata = self.xml2jsonl(xml_file)
            if not os.path.exists(processed_name):
                print(f"Can't find file {processed_name}")
                continue
            img_metadata.append(metadata)

        return img_metadata

    def construct_hf_dataset(self) -> Dataset:
        return Dataset.from_list(self._load_metadata()).cast_column("image", Image())

    def get_user_prompt(self):
        prompt = ("Count the number of trees in the given image to the best of your ability. Output your count only "
                  "without any further explanation.")

        return prompt

    def query_gpt4(self):
        hf_dataset = load_dataset("danielz01/neon-trees", split=self.split)
        result_path = os.path.join(self.dataset_base, f"gpt-4v-counting.jsonl")

        final_results = resume_from_jsonl(result_path)
        for i, data_item in enumerate(hf_dataset):
            if any([data_item["path"] == x["path"] for x in final_results]):
                print(f'Skipping {data_item["path"]}')
                continue

            png_buffer = io.BytesIO()

            # We save the image in the PNG format to the buffer.
            data_item["image"].save(png_buffer, format="PNG")

            image_base64 = base64.b64encode(png_buffer.getvalue()).decode('utf-8')
            user_prompt = self.get_user_prompt()
            payload, response = self.query_openai(image_base64, system=self.system_message, user=user_prompt)
            print(data_item["path"], response)

            data_item.pop("image")
            final_results.append({
                "index": i,
                **data_item,
                "response": response
            })

            dump_to_jsonl(final_results, result_path)


def main():
    dataset = NeonTreeEvaluationDataset(split="evaluation", credential_path="./.secrets/openai.json")
    print(dataset.system_message)
    print(dataset.get_user_prompt())
    dataset.query_gpt4()


def evaluation(result_path, ax=None):
    *model, city = os.path.basename(result_path).removesuffix(".jsonl").split("-")
    model_name = "-".join(model)

    result_json = pd.read_json(result_path, lines=True)
    if "gpt" in os.path.basename(result_path).lower():
        result_json["model_response"] = result_json["response"].apply(lambda x: x["choices"][0]["message"]["content"])
    else:
        result_json["model_response"] = result_json["response"]
    if "count" not in result_json.columns:
        result_json["objects"] = pd.read_json(
            os.path.join(os.path.dirname(result_path), "gpt-4v-counting.jsonl"), lines=True
        )["objects"]

    result_json["parsed_response"] = result_json["model_response"].apply(parse_digit_response)
    result_json["count"] = result_json["objects"].apply(lambda x: len(x["bbox"]))
    result_json_no_refusal = result_json[result_json["parsed_response"] != -1].copy()
    result_json_refusal = result_json[result_json["parsed_response"] == -1].copy()

    rr, (mape, mape_no_refusal), (mae, mae_no_refusal), (r2, r2_no_refusal) = calculate_counting_metrics(result_json, result_json_no_refusal)

    print(os.path.basename(result_path))
    print(f"MAE & MAE (No Refusal) & MAPE & MAPE (No Refusal) & R2: {r2:.4f} & R2 (No Refusal) & Refusal Rate")
    print(f"{mae:.4f} & {mae_no_refusal:.4f} & {mape:.4f} & {mape_no_refusal:.4f} & {r2:.4f} & {r2_no_refusal:.4f} & "
          f"{rr:.4f}")

    print(f"MAE (No Refusal) & MAPE (No Refusal) & R2 (No Refusal) & Refusal Rate")
    print(f"{mae_no_refusal:.3f} & {mape_no_refusal:.3f} & {r2_no_refusal:.3f} & {rr:.2f}")

    plot_scatter(result_json_no_refusal, ax=ax)
    if ax:
        ax.set_title(model_name)

    plt.savefig(result_path.replace(".jsonl", ".pdf"))
    plt.savefig(result_path.replace(".jsonl", ".png"))


if __name__ == "__main__":
    files = [
        "./data/NeonTreeEvaluation/gpt-4v-counting.jsonl",
        "./data/NeonTreeEvaluation/Qwen-VL-Chat-counting.jsonl",
        "./data/NeonTreeEvaluation/instructblip-flan-t5-xxl-counting.jsonl",
        "./data/NeonTreeEvaluation/instructblip-vicuna-13b-counting.jsonl",
        "./data/NeonTreeEvaluation/llava-v1.5-13b-counting.jsonl",
    ]
    for file in files:
        evaluation(file)
