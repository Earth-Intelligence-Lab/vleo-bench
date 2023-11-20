import json
import os
from glob import glob
from typing import Dict, List

import numpy as np
from datasets import Dataset, Image
from torchgeo.datasets import XView2
from shapely.wkt import loads as loads_wkt
from shapely.geometry import mapping, Polygon
from src.datasets.dataset import VLEODataset


def get_int_coords(x):
    return np.array(x).round().astype(np.int32)


class XView2CompetitionDataset(XView2, VLEODataset):
    def __init__(self, root: str = "data", split: str = "train"):
        super().__init__(root, split)

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


def main():
    for split in ["train", "test"]:
        dataset = XView2CompetitionDataset("datasets/xView2/", split=split)
        hf_dataset = dataset.construct_hf_dataset()
        print(hf_dataset)
        hf_dataset.push_to_hub("danielz01/xView2", "competition", split=split)


if __name__ == "__main__":
    main()
