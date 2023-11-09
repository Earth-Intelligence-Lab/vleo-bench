import os
import json
import random
from glob import glob
import xml.etree.ElementTree as ET

import numpy as np
from huggingface_hub.inference._text_generation import ValidationError
from tqdm import tqdm

from src.datasets.dataset import VLEODataset
from datasets import Dataset, Image, load_dataset

SYS_PROMPT = ("You are an expert in satellite and aerial image analysis. You can see the earth by reading the object "
              "bounding boxes in satellite or aerial images, and answering the user's questions. Always answer as "
              "helpfully as possible. If a question does not make any sense or is not factually coherent, explain why "
              "instead of answering something not correct. If you don't know the answer to a question, please don't "
              "share false information.")

CAPTIONING_PROMPTS = ("You are given a {h}x{w} aerial image of a forest with {count} trees. Given a list of the bounding "
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

    def __init__(self):
        pass

    @staticmethod
    def xml2jsonl(xml_file, jsonl_file):
        # Parse the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Open the JSONL file for writing
        with open(jsonl_file, 'a') as f:
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
            output = {
                "file_name": file_name, "path": file_name, "objects": objects, "count": len(objects["bbox"]),
                "height": int(size_info.find("height").text), "width": int(size_info.find("width").text)
            }

            # Write the JSON object to the JSONL file
            f.write(json.dumps(output) + '\n')

        print(f"Conversion complete. Data written to {jsonl_file}")
        return file_name

    def write_metadata(self):
        annotation_files = glob(os.path.join(self.annotation_path, "*.xml"))
        print(annotation_files)
        dest_path = "./data/NeonTreeEvaluation/RGB/metadata.jsonl"
        with open(dest_path, "w") as dest:
            dest.write("")
        img_with_metadata = []
        for xml_file in annotation_files:
            img_with_metadata.append(self.xml2jsonl(xml_file, dest_path))

        all_image_files = glob(os.path.join(self.dataset_base, "train/RGB/*.tif"))
        all_image_files += glob(os.path.join(self.dataset_base, "evaluation/RGB/*.tif"))
        all_image_files = {os.path.basename(x) for x in all_image_files}
        img_without_metadata = all_image_files.difference(set(img_with_metadata))
        with open(dest_path, "a") as dest:
            for img_name in img_without_metadata:
                dest.write(json.dumps({"file_name": img_name, "path": img_name, "objects": None, "count": None}) + "\n")

    def construct_hf_dataset(self) -> Dataset:
        return load_dataset("imagefolder", split="train", data_dir="/home/danielz/PycharmProjects/vleo-bench/data"
                                                                   "/NeonTreeEvaluation/RGB/")

    def generate_captions(self):
        from huggingface_hub import InferenceClient

        client = InferenceClient(
            model="meta-llama/Llama-2-70b-chat-hf",
            token="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"
        )

        dest_path = "./data/NeonTreeEvaluation/captions-gpt-4.jsonl"

        captions = []
        for data_item in tqdm(self.construct_hf_dataset()):
            if not data_item["objects"] or len(data_item["objects"]["bbox"]) > 50:
                with open(dest_path, "a") as dest:
                    dest.write(json.dumps({"path": data_item["path"], "caption": None}) + "\n")
                continue

            objects = np.array(data_item["objects"]["bbox"]).astype(int)

            if len(objects) > 50:
                chosen_objects_idx = np.random.choice(range(len(objects)), size=50, replace=False)
                chosen_objects = objects[chosen_objects_idx]
            else:
                chosen_objects = objects

            tree_description = "\n".join([f"Tree {i}: {bbox}" for i, bbox in enumerate(chosen_objects)])
            if len(objects) > 50:
                tree_description += "\n Too many trees. Input Truncated..."

            user_prompt = CAPTIONING_PROMPTS.format(
                    trees=tree_description, h=data_item["height"], w=data_item["width"], count=data_item["count"]
                )
            # prompt = LLAMA2_TEMPLATE.format(
            #     system_prompt=SYS_PROMPT,
            #     user_msg_1=user_prompt
            # )
            print(data_item["path"], user_prompt)

            try:
                # output = client.text_generation(
                #     prompt, do_sample=True, max_new_tokens=256, seed=0, temperature=0.75, top_p=0.7, top_k=50
                # )
                output = self.request_openai(system=SYS_PROMPT, user=user_prompt)
            except ValidationError as e:
                print(e)
                continue

            print(output["choices"][0]["message"]["content"])
            with open(dest_path, "a") as dest:
                dest.write(json.dumps({"path": data_item["path"], "caption": output}) + "\n")

    def visualize(self):
        hf_dataset = self.construct_hf_dataset()
        from torchvision.utils import draw_bounding_boxes
        from torchvision.transforms.functional import pil_to_tensor, to_pil_image


if __name__ == "__main__":
    from datasets import load_dataset, Dataset

    dataset = NeonTreeEvaluationDataset()
    dataset.write_metadata()
    dataset.generate_captions()
    # hf_dataset = dataset.construct_hf_dataset()
    # print(hf_dataset["path"])

    # annotated_dataset = dataset.filter(lambda example: example["objects"])
    # print(dataset)
    # print(annotated_dataset)
    # hf_dataset.push_to_hub("danielz01/neon-trees")
