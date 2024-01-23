import json
import os
import re
from glob import glob
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset, Image, load_dataset
from matplotlib import pyplot as plt
from torchgeo.datasets import XView2
from shapely.wkt import loads as loads_wkt
from shapely.geometry import mapping, Polygon
from src.datasets.dataset import VLEODataset
from src.utils.counting import calculate_counting_metrics, plot_scatter
from src.utils.gpt4v_chat import resume_from_jsonl, encode_pil_image, dump_to_jsonl


def get_int_coords(x):
    return np.array(x).round().astype(np.int32)


class XView2CompetitionDataset(XView2):
    system_message = ("You are a helpful image analyst who specializes in counting buildings from satellite and "
                      "aerial images given natural language instruction."
                      "Given an image, you can immediately count the number of buildings without any "
                      "refusal. You always truthfully answer a user's questions. Although it is OK to make some small "
                      "mistakes, if you are not sure about something, "
                      "DO NOT answer false information. Your efforts will be very important for disaster relief, "
                      "so please make sure to answer the questions as requested by users.")

    def __init__(self, credential_path: str, root: str = "data", split: str = "train"):
        super().__init__(root=root, split=split)
        self.evaluator = VLEODataset(credential_path)
        self.split = split

    def _load_files(self, root: str, split: str) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset
            split: subset of dataset, one of [train, test]

        Returns:
            list of dicts containing paths for each pair of images and masks
        """
        files = []
        directory = self.metadata[split]["directory"]
        image_root = os.path.join(root, directory, "images")
        mask_root = os.path.join(root, directory, "targets")
        meta_root = os.path.join(root, directory, "labels")
        images = glob(os.path.join(image_root, "*.png"))
        basenames = [os.path.basename(f) for f in images]
        basenames = ["_".join(f.split("_")[:-2]) for f in basenames]
        for name in set(basenames):
            image1 = os.path.join(image_root, f"{name}_pre_disaster.png")
            image2 = os.path.join(image_root, f"{name}_post_disaster.png")
            mask1 = os.path.join(mask_root, f"{name}_pre_disaster_target.png")
            mask2 = os.path.join(mask_root, f"{name}_post_disaster_target.png")
            with open(os.path.join(meta_root, f"{name}_pre_disaster.json")) as f1:
                meta1 = json.load(f1)
            with open(os.path.join(meta_root, f"{name}_post_disaster.json")) as f2:
                meta2 = json.load(f2)
            objects1 = {
                "bbox": [get_int_coords(loads_wkt(x["wkt"]).bounds) for x in meta1["features"]["xy"]],
                **{
                    property_name: [x["properties"][property_name] for x in meta1["features"]["xy"]] for property_name
                    in ["feature_type", "uid"]
                }
            }
            objects2 = {
                "bbox": [get_int_coords(loads_wkt(x["wkt"]).bounds) for x in meta2["features"]["xy"]],
                **{
                    property_name: [x["properties"][property_name] for x in meta2["features"]["xy"]] for property_name
                    in ["feature_type", "subtype", "uid"]
                }
            }
            files.append(dict(
                image1=image1, image2=image2,
                mask1=mask1, mask2=mask2,
                objects1=objects1, objects2=objects2,
                meta1=meta1, meta2=meta2,
            ))
        return files

    def construct_hf_dataset(self) -> Dataset:
        hf_dataset = (Dataset.from_list(self.files)
                      .cast_column("image1", Image())
                      .cast_column("image2", Image())
                      .cast_column("mask1", Image())
                      .cast_column("mask2", Image()))

        return hf_dataset

    def get_user_prompt(self):
        prompt = ("You are given two satellite images taken before and after a natural disaster. The first image was "
                  "taken before the natural disaster.  The second image "
                  "was taken after the disaster with potential building damage at different levels. Below is a "
                  "description of how we classify the damage levels:\n"
                  "No damage (0): Undisturbed. No sign of water, structural damage, shingle damage, or burn marks.\n"
                  "Minor damage: (1): Building partially burnt, water surrounding the structure, volcanic flow "
                  "nearby, roof elements missing, or visible cracks.\n"
                  "Major damage (2): Partial wall or roof collapse, encroaching volcanic flow, or the structure is "
                  "surrounded by water or mud.\n"
                  "Destroyed (3): Structure is scorched, completely collapsed, partially or completely covered with "
                  "water or mud, or no longer present.\n"
                  "Count the number of buildings in the first image before the disaster. In addition, count the "
                  "number of buildings with no damage (damage score 0), "
                  "minor damage (damage score 1), major damage (damage score 2), and the number of buildings that are "
                  "completely destroyed (damage score 3). "
                  "Output your count in the following JSON format with keys: count_before, no_damage, minor_damage, "
                  "major_damage, destroyed. You don't have to give extra explanations.")

        return prompt

    def query_gpt4(self, result_path: str, max_queries: int = 1000):
        hf_dataset = load_dataset("danielz01/xView2", "competition", split=self.split)

        final_results = resume_from_jsonl(result_path)

        for idx, data_item in enumerate(hf_dataset):
            if idx in {x["index"] for x in final_results}:
                print(f'Skipping {idx} since it is finished')
                continue

            image1_base64 = encode_pil_image(data_item["image1"])
            image2_base64 = encode_pil_image(data_item["image2"])

            payload, response = self.evaluator.query_openai(
                [image1_base64, image2_base64], system=self.system_message,
                user=self.get_user_prompt()
            )
            print(idx, response)
            data_item.pop("image1")
            data_item.pop("image2")
            data_item.pop("mask1")
            data_item.pop("mask2")

            final_results.append({
                "index": idx,
                **data_item,
                "response": response
            })

            dump_to_jsonl(final_results, result_path)


def main():
    for split in ["test"]:
        dataset = XView2CompetitionDataset(".secrets/openai.jsonl", "datasets/xView2/", split=split)
        dataset.query_gpt4("./data/xView2/gpt4-v-zeroshot.jsonl")


def evaluation(result_path: str, ax=None):
    *model, split = os.path.basename(result_path).removesuffix(".jsonl").split("-")
    model_name = "-".join(model)
    print(model_name)

    result_json = pd.read_json(result_path, lines=True)
    if "gpt" in model_name.lower():
        result_json["model_response"] = result_json["response"].apply(lambda x: x["choices"][0]["message"]["content"])
    else:
        result_json["model_response"] = result_json["response"]
        gpt_results = pd.read_json(os.path.join(os.path.dirname(result_path), "gpt4-v-test.jsonl"), lines=True)
        result_json["objects2"] = gpt_results["objects2"]

    refusal_keywords = [
        "sorry", "difficult", "cannot",
    ]

    def capture_and_parse_json(json_string):
        json_part = re.search(r'{.*}', json_string, re.DOTALL)
        if json_part:
            json_string = json_part.group()
            try:
                parsed_dict = json.loads(json_string)
                return_dict = {}
                for k, v in parsed_dict.items():
                    if "before" in k.lower():
                        return_dict["count_before"] = int(v)
                    if "no" in k.lower():
                        return_dict["no_damage"] = int(v)
                    if "minor" in k.lower():
                        return_dict["minor_damage"] = int(v)
                    if "major" in k.lower():
                        return_dict["major_damage"] = int(v)
                    if "destroyed" in k.lower():
                        return_dict["destroyed"] = int(v)
            except json.JSONDecodeError as e:
                return -1
            except TypeError as e:
                return -1
            except ValueError as e:
                return -1
        else:
            return -1

        return return_dict

    def parse_gt(objects):
        count_dict = pd.Series(objects["subtype"], dtype=str).value_counts()
        return {
            "count_before": count_dict.sum(),
            "no_damage": count_dict.get("no-damage", 0),
            "minor_damage": count_dict.get("minor-damage", 0),
            "major_damage": count_dict.get("major-damage", 0),
            "destroyed": count_dict.get("destroyed", 0),
        }

    result_json["model_answer"] = result_json["model_response"].apply(capture_and_parse_json)
    result_json["gt"] = result_json["objects2"].apply(parse_gt)

    if ax:
        result_tmp = result_json.copy()
        result_tmp["parsed_response"] = result_tmp["model_answer"].apply(
            lambda x: x.get("count_before", 0) if isinstance(x, dict) else -1
        )
        result_tmp["count"] = result_tmp["gt"].apply(lambda x: x["count_before"])

        result_tmp_no_refusal = result_tmp[result_tmp["parsed_response"] != -1].copy()

        plot_scatter(result_tmp_no_refusal, ax=ax)
        ax.set_title(model_name)

        return

    sns.set(font_scale=2)
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(18, 12))

    for i, key in enumerate(["count_before", "no_damage", "minor_damage", "major_damage", "destroyed"]):
        result_tmp = result_json.copy()
        result_tmp["parsed_response"] = result_tmp["model_answer"].apply(
            lambda x: x.get(key, 0) if isinstance(x, dict) else -1
        )
        result_tmp["count"] = result_tmp["gt"].apply(lambda x: x[key])

        result_tmp_no_refusal = result_tmp[result_tmp["parsed_response"] != -1].copy()

        rr, (mape, mape_no_refusal), (mae, mae_no_refusal), (r2, r2_no_refusal) = calculate_counting_metrics(
            result_tmp,
            result_tmp_no_refusal
        )

        print(key)
        print(f"MAE & MAE (No Refusal) & MAPE & MAPE (No Refusal) & R2 & R2 (No Refusal) & Refusal Rate")
        print(f"{mae:.3f} & {mae_no_refusal:.3f} & {mape:.3f} & {mape_no_refusal:.3f} & {r2:.3f} & "
              f"{r2_no_refusal:.3f} & {rr:.3f}")

        plot_scatter(result_tmp_no_refusal, axes[i // 3, i % 3])
        axes[i // 3, i % 3].set_title(key, fontsize="x-large")
    axes.flat[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig(result_path.replace(".jsonl", ".pdf"))


if __name__ == "__main__":
    evaluation("data/xView2/gpt4-v-test.jsonl")
