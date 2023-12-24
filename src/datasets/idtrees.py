from typing import Optional, Callable

import geopandas
import numpy as np
import pandas as pd
from datasets import Dataset, ClassLabel, Image
from torch import Tensor
from PIL import Image as PILImage
from torchgeo.datasets import IDTReeS
from src.datasets.dataset import VLEODataset


class IDTreesDataset(IDTReeS):
    train_labels = "./data/IDTReeS/train/Field/train_data.csv"
    train_shps = [
        "./data/IDTReeS/train/ITC/train_MLBS.shp",
        "./data/IDTReeS/train/ITC/train_OSBS.shp"
    ]
    crown2rs = "./data/IDTReeS/train/Field/itc_rsFile.csv"
    join_key = "indvdID"

    def join_shapefiles(self):
        train_labels = pd.read_csv(self.train_labels)
        train_rs_files = pd.read_csv(self.crown2rs)
        train_labels = pd.merge(
            left=train_labels,
            right=train_rs_files.drop("id", axis=1),
            on=self.join_key, how="left"
        )

        train_shps = pd.concat([geopandas.read_file(filename) for filename in self.train_shps])
        train_shps_meta = geopandas.GeoDataFrame(
            pd.merge(left=train_shps, right=train_labels, on=self.join_key, how="left")
        )
        train_shps_meta.to_file(self.train_labels.replace(".csv", ".gpkg"), driver="GPKG")

    def _hf_item_generator(self):
        for idx in range(len(self)):
            try:
                data_item = self[idx]
            except IndexError or TypeError as e:
                print(e)
                continue

            img = data_item["image"].numpy().astype(np.uint8).transpose(1, 2, 0)
            image = PILImage.fromarray(img)
            labels = data_item.get("label", [-1])
            bbox = data_item.get("boxes", [[]]),

            data_item.pop("las")
            data_item.pop("chm")
            data_item.pop("hsi")
            data_item.pop("image")
            data_item.pop("label", [])

            objects = {
                "bbox": bbox,
                "categories": labels
            }
            yield {
                "image": image,
                "objects": objects,
                "taxonomy_id": [self.idx2class[i.item()] if i != -1 else "" for i in labels],
                "scientific_name": [self.classes[self.idx2class[i.item()]] if i != -1 else "" for i in labels],
                **data_item
            }

    def construct_hf_dataset(self) -> Dataset:
        return Dataset.from_generator(self._hf_item_generator)

    def query_gpt4(self):
        hf_dataset = self.construct_hf_dataset()


def main():
    dataset = IDTreesDataset(root="./data/IDTReeS/", split="train")
    hf_dataset = dataset.construct_hf_dataset()
    print(hf_dataset)
    hf_dataset.push_to_hub("danielz01/IDTreeS", split="train")


if __name__ == "__main__":
    main()
