import re
import json
import os.path
import string
from glob import glob

import pandas as pd
from datasets import Dataset, Image

from src.datasets.dataset import VLEODataset


class RSICDDataset(VLEODataset):
    dir_name = "RSICD"
    image_dir = "RSICD_images"
    class_dir = "txtclasses_rsicd"
    caption_fname = "dataset_rsicd.json"
    caption_splits = ["train", "val"]

    CAPTION_KEY = "images"

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = os.path.join(data_dir, self.dir_name)
        self.caption_path = os.path.join(self.data_dir, self.caption_fname)
        with open(self.caption_path) as f:
            self.captions = json.load(f)[self.CAPTION_KEY]

        self.land_covers = {}
        for land_cover_file in glob(os.path.join(data_dir, self.dir_name, self.class_dir, "*.txt")):
            land_cover_type = os.path.basename(land_cover_file).replace(".txt", "")
            land_cover_type = string.capwords(re.sub("([^-])([A-Z][a-z-]+)", r"\1 \2", land_cover_type))
            with open(land_cover_file) as f:
                for line in f.readlines():
                    img_file = line.strip()
                    if img_file in self.land_covers:
                        self.land_covers[img_file].append(land_cover_type)
                    else:
                        self.land_covers[img_file] = [land_cover_type]

    def convert_hf_dataset(self, split: str) -> "Dataset":
        assert split in self.caption_splits
        captions = list(filter(lambda x: x["split"] == split, self.captions))
        assert len(captions) > 0
        hf_dicts = []
        for caption in captions:
            dataset_entry = {
                "image": os.path.join(self.data_dir, self.image_dir, caption["filename"]),
                "path": caption["filename"],
                "img_id": caption["imgid"],
                "captions": [x["raw"] for x in caption["sentences"]],
                "caption_ids": [x["sentid"] for x in caption["sentences"]]
            }
            hf_dicts.append(dataset_entry)

        result_dataset = Dataset.from_pandas(pd.DataFrame(hf_dicts)).cast_column("image", Image())
        return result_dataset


if __name__ == "__main__":
    dataset = RSICDDataset()
    for split_name in ["train", "val"]:
        hf_dataset = dataset.convert_hf_dataset(split_name)
        # hf_dataset.push_to_hub("danielz01/rsicd", split=split_name)
        print(hf_dataset[0])
