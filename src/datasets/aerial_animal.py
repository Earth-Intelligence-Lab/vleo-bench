import os

import pandas as pd
from datasets import Dataset, Image

from src.datasets.dataset import VLEODataset


class AerialAnimalPopulation4TuDataset(VLEODataset):
    directory_name = "aerial-animal-population-4tu"

    splits = ["train", "val", "test"]
    annotation_header = ["path", "xmin", "ymin", "xmax", "ymax", "species"]

    spices = ['Elephant', 'Giraffe', 'Zebra']

    def __init__(self, root: str = "./data", split: str = "test"):
        assert split in self.splits

        self.root = root
        self.split = split

        self.directory = os.path.join(root, self.directory_name)

        metadata = pd.read_csv(
            os.path.join(self.directory, f"annotations_{self.split}.csv"),
            names=self.annotation_header
        )
        if self.split == "train":
            metadata = metadata[["MVER" not in x for x in metadata["path"]]]
            metadata.reset_index(drop=True)
        metadata_grouped = metadata.groupby('path').agg({
            'xmin': list,
            'ymin': list,
            'xmax': list,
            'ymax': list,
            'species': list  # Now taking all species into account
        }).reset_index()

        # Create the desired nested list format for 'bbox'
        metadata_grouped['objects'] = metadata_grouped.apply(lambda row: {
            "bbox": list(zip(row['xmin'], row['ymin'], row['xmax'], row['ymax'])),
            "categories": row['species']
        }, axis=1)
        metadata_grouped["image"] = metadata_grouped["path"].apply(lambda x: os.path.join(self.directory, x))

        self.metadata = metadata_grouped[["image", "objects", "path"]]

    def convert_to_hf_dataset(self) -> Dataset:
        hf_dataset = Dataset.from_pandas(self.metadata).cast_column("image", Image())

        return hf_dataset


def main():
    for split in ["train"]:  # , "val", "test"]:
        dataset = AerialAnimalPopulation4TuDataset(split=split)
        hf_dataset = dataset.convert_to_hf_dataset()
        print(dataset.metadata)
        print(hf_dataset)


if __name__ == "__main__":
    main()
