import base64
import io
import os
import json
import random
from glob import glob
import xml.etree.ElementTree as ET

import pandas as pd
from PIL import Image as PILImage

import numpy as np
from huggingface_hub.inference._text_generation import ValidationError
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm

from src.datasets.dataset import VLEODataset
from datasets import Dataset, Image, load_dataset

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

    def __init__(self, split: str = "evaluation"):
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

    def generate_captions(self):
        from huggingface_hub import InferenceClient

        client = InferenceClient(
            model="meta-llama/Llama-2-70b-chat-hf",
            token=os.environ["HF_TOKEN"]
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
    dataset = NeonTreeEvaluationDataset(split="evaluation")
    print(dataset.system_message)
    print(dataset.get_user_prompt())
    dataset.query_gpt4()


def evaluation(result_path):
    result_json = pd.read_json(result_path, lines=True)
    result_json["model_response"] = result_json["response"].apply(lambda x: x["choices"][0]["message"]["content"])

    def parse_response(x):
        try:
            ret = int(x)
        except ValueError:
            ret = -1
        return ret

    result_json["parsed_response"] = result_json["model_response"].apply(parse_response)
    result_json["count"] = result_json["objects"].apply(lambda x: len(x["bbox"]))
    result_json_no_refusal = result_json[result_json["parsed_response"] != -1]
    result_json_refusal = result_json[result_json["parsed_response"] == -1]

    rr = (result_json["parsed_response"] == -1).mean()

    mape = mean_absolute_percentage_error(
        y_true=result_json["count"],
        y_pred=result_json["parsed_response"].replace(-1, 0)
    )
    mape_no_refusal = mean_absolute_percentage_error(
        y_true=result_json_no_refusal["count"],
        y_pred=result_json_no_refusal["parsed_response"]
    )

    r2 = np.corrcoef(result_json["count"], result_json["parsed_response"].replace(-1, 0))[0, 1] ** 2
    r2_no_refusal = np.corrcoef(
        result_json_no_refusal["count"], result_json_no_refusal["parsed_response"].replace(-1, 0)
    )[0, 1] ** 2

    print(os.path.basename(result_path))
    print(f"Refusal: {rr:.4f} \t | MAPE: {mape:.4f} | MAPE (No Refusal): {mape_no_refusal:.4f} | R2: {r2:.4f} | R2 ("
          f"No Refusal): {r2_no_refusal:.4f}")
    print(result_json_refusal)


if __name__ == "__main__":
    evaluation("/home/danielz/PycharmProjects/vleo-bench/data/NeonTreeEvaluation/gpt-4v-counting.jsonl")
