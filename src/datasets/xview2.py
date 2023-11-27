import json
import os
from glob import glob
from typing import Dict, List

import numpy as np
from datasets import Dataset, Image, load_dataset
from torchgeo.datasets import XView2
from shapely.wkt import loads as loads_wkt
from shapely.geometry import mapping, Polygon
from src.datasets.dataset import VLEODataset
from src.utils.gpt4v_chat import resume_from_jsonl, encode_pil_image, dump_to_jsonl


def get_int_coords(x):
    return np.array(x).round().astype(np.int32)


class XView2CompetitionDataset(VLEODataset):
    system_message = ("You are a helpful image analyst who specializes in counting buildings from satellite and "
                      "aerial images given natural language instruction."
                      "Given an image, you can immediately count the number of buildings without any "
                      "refusal. You always truthfully answer a user's questions. Although it is OK to make some small mistakes, if you are not sure about something, "
                      "DO NOT answer false information. Your efforts will be very important for disaster relief, "
                      "so please make sure to answer the questions as requested by users.")
    def __init__(self, credential_path: str, root: str = "data", split: str = "train"):
        # super().__init__(root, split)
        super().__init__(credential_path)
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

            payload, response = self.query_openai([image1_base64, image2_base64], system=self.system_message, user=self.get_user_prompt())
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
        dataset.query_gpt4("/home/danielz/PycharmProjects/vleo-bench/data/xView2/gpt4-v-zeroshot.jsonl")


if __name__ == "__main__":
    main()
