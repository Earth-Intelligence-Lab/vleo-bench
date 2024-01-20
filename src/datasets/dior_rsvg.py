import os.path
import xml.etree.ElementTree as ET
from typing import List

import pandas as pd
from termcolor import colored, cprint
import numpy as np
from datasets import Dataset, Image
import torch
from torchvision.ops import box_iou
from src.datasets.dataset import VLEODataset
from src.utils.detection import extract_bbox, get_mean_centroid_distance, extract_qwen_bbox
from src.utils.gpt4v_chat import resume_from_jsonl, encode_image, dump_to_jsonl


def get_file_list(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name, files in os.walk(root) for f in files if
            f.endswith(file_type)]


class DIORRSVGDataset(VLEODataset):
    folder_name = "DIOR-RSVG"
    jpeg_directory = os.path.join(folder_name, "JPEGImages")
    annotation_directory = os.path.join(folder_name, "Annotations")
    category_name = {
        'basketballcourt': 'Basketball Court', 'overpass': 'Overpass', 'dam': 'Dam',
        'groundtrackfield': 'Ground Track Field', 'storagetank': 'Storage Tank', 'stadium': 'Stadium',
        'airplane': 'Airplane', 'golffield': 'Golf Field', 'baseballfield': 'Baseball Field',
        'Expressway-Service-area': 'Expressway Service Area', 'bridge': 'Bridge', 'tenniscourt': 'Tennis Court',
        'harbor': 'Harbor', 'windmill': 'Windmill', 'ship': 'Ship', 'chimney': 'Chimney', 'airport': 'Airport',
        'Expressway-toll-station': 'Expressway Toll Station', 'trainstation': 'Train Station', 'vehicle': 'Vehicle'
    }

    system_message = ("You are a helpful image analyst that specializes in localizing objects from satellite "
                      "and aerial images given a natural language instruction. "
                      "You always truthfully answer the user's question. If you are not sure about "
                      "something, don't answer false information.")

    def __init__(self, credential_path: str, root: str = "./data", split: str = "test"):
        super().__init__(credential_path)
        assert split in ["train", "test", "val"]

        self.root = root
        self.split = split

        self.object_list = self._get_objects()
        self.categories = sorted(list({x["object"]["categories"] for x in self.object_list}))

    def _get_objects(self):
        count = 0
        object_list = []
        annotations = get_file_list(os.path.join(self.root, self.annotation_directory), ".xml")
        with open(os.path.join(self.root, self.folder_name, f"{self.split}.txt"), "r") as f:
            split_index = [int(x.strip()) for x in f.readlines()]
        for annotation_path in annotations:
            root = ET.parse(annotation_path).getroot()
            for member in root.findall('object'):
                if count in split_index:
                    img_file = os.path.join(self.root, self.jpeg_directory, root.find("./filename").text)
                    bbox = np.array([
                        int(member[2][0].text), int(member[2][1].text),
                        int(member[2][2].text), int(member[2][3].text)
                    ], dtype=np.int32)
                    class_name = member[0].text
                    text = member[3].text

                    object_list.append({
                        "image": img_file,
                        "object": {
                            "bbox": bbox, "categories": class_name,
                            "categories_normalized": self.category_name[class_name], "captions": text}
                    })
                count += 1

        return object_list

    def _get_images(self):
        images = {x["image"] for x in self.object_list}
        image_dict = {
            x: {
                "path": os.path.basename(x),
                "objects": {"bbox": [], "categories": [], "categories_normalized": [], "captions": []}
            } for x in images
        }
        for bbox_object in self.object_list:
            for k, v in bbox_object["object"].items():
                image_dict[bbox_object["image"]]["objects"][k].append(v)
        image_list = [{"image": k, **v} for k, v in image_dict.items()]

        return image_list

    def construct_hf_dataset(self) -> Dataset:
        return Dataset.from_list(self._get_images()).cast_column("image", Image())

    @staticmethod
    def get_detection_prompt(h: int, w: int, description: str):
        prompt = (f"You are given an {h} x {w} satellite image. Identify the extent of the object in the description "
                  "below in the format of [xmin, ymin, xmax, ymax], where the top-left coordinate is (x_min, "
                  "y_min) and the bottom-right coordinate is (x_max, y_max). You should answer the extent without "
                  f"further explanation.\nDescription: {description}")
        return prompt

    def get_text_bbox_classification_prompt(self, h: int, w: int, bbox: List[int]):
        categories_normalized = sorted(self.category_name.values())
        xmin, ymin, xmax, ymax = bbox
        prompt = (f"Given a {h} x {w} satellite image and a bounding box of an object in the format of [xmin, ymin, "
                  f"xmax, ymax], where the top-left coordinate is (x_min, y_min) and the bottom-right coordinate is ("
                  f"x_max, y_max). Classify the type of the object in the bounding box into one of the following "
                  f"categories.\nCategories:\n")
        prompt += ";\n".join([f"{i + 1}. {option}" for i, option in enumerate(categories_normalized)])
        prompt += f"\nObject Location in Bounding Box: [{xmin}, {ymin}, {xmax}, {ymax}]\n"
        prompt += "Think step by step and output your final answer in the last line."

        return prompt

    def get_segmentation_demonstrations(self):
        demonstrations = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_image(os.path.join(self.root, self.jpeg_directory, '14955.jpg'))}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": self.get_detection_prompt(800, 800, "A windmill on the left")
                    },
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "[100, 342, 164, 439]"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_image(os.path.join(self.root, self.jpeg_directory, '10365.jpg'))}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": self.get_detection_prompt(800, 800, "A large airplane")
                    },
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "[172, 368, 434, 577]"
                    }
                ]
            }
        ]

        return demonstrations

    def query_gpt4(self, result_path: str, max_queries: int = 100, few_shot=False):
        hf_dataset = self.construct_hf_dataset()

        np.random.seed(0)
        selected_indices = np.random.choice(range(len(hf_dataset)), size=max_queries, replace=False).astype(int).tolist()
        final_results = resume_from_jsonl(result_path)
        for idx in selected_indices:
            data_item = hf_dataset[int(idx)]
            if any([idx == x["index"] for x in final_results]):
                print(f'Skipping {idx}')
                continue

            image_base64 = encode_image(os.path.join(self.root, self.jpeg_directory, data_item["path"]))
            payload, response = self.query_openai(
                image_base64,
                system=self.system_message,
                user=self.get_text_bbox_classification_prompt(
                    data_item["image"].height,
                    data_item["image"].width,
                    data_item["objects"]["bbox"][0]
                ),
                # demos=self.get_segmentation_demonstrations() if few_shot else []
            )
            cprint(f'{idx} {response["choices"][0]["message"]["content"]}', "blue")
            data_item.pop("image")

            final_results.append({
                "index": idx,
                "object": {k: v[0] for k, v in data_item["objects"].items()},
                "response": response,
                "path": data_item["path"]
            })

            dump_to_jsonl(final_results, result_path)


def main():
    dataset = DIORRSVGDataset(credential_path=".secrets/openai_1700801522.jsonl", split="test")
    dataset.query_gpt4("./data/DIOR-RSVG/gpt-4v-detection-text-bbox.jsonl", max_queries=1000, few_shot=False)


def evaluation(result_path: str):
    *model, split = os.path.basename(result_path).removesuffix(".jsonl").split("-")
    model_name = "-".join(model)
    print(model_name)

    result_json = pd.read_json(result_path, lines=True)
    if "gpt" in model_name.lower():
        result_json["model_response"] = result_json["response"].apply(lambda x: x["choices"][0]["message"]["content"])
    else:
        result_json["model_response"] = result_json["response"]
        gpt_results = pd.read_json(os.path.join(os.path.dirname(result_path), "gpt-4v-segmentation-zeroshot.jsonl"), lines=True)
        result_json["object"] = gpt_results["object"]

    if "qwen" in model_name.lower():
        result_json["model_answer"] = result_json["model_response"].apply(extract_qwen_bbox)
    else:
        result_json["model_answer"] = result_json["model_response"].apply(extract_bbox)
    result_json["gt"] = result_json["object"].apply(lambda x: x["bbox"])

    result_json["is_refusal"] = result_json["model_answer"].apply(lambda x: len(x) == 0)
    rr = result_json["is_refusal"].mean()
    print(rr)

    result_json_no_refusal = result_json[~result_json["is_refusal"]].copy()
    result_json_no_refusal["model_answer"] = result_json_no_refusal["model_answer"].apply(
        lambda x: np.array(x).reshape(-1)
    )
    predicted_bbox = np.stack(result_json_no_refusal["model_answer"].to_list())
    gt_bbox = np.stack(result_json_no_refusal["gt"].to_list())
    ious = torch.diag(box_iou(torch.from_numpy(predicted_bbox), torch.from_numpy(gt_bbox)))

    mean_iou = ious.mean().item()
    pr_05 = torch.mean((ious > 0.5).float()).item()
    mean_dist = get_mean_centroid_distance(predicted_bbox, gt_bbox)

    print(mean_iou, pr_05, mean_dist, rr)


if __name__ == "__main__":
    evaluation("./data/DIOR-RSVG/Qwen-VL-Chat-test-det.jsonl")
