import os
import pandas as pd
from datasets import Dataset, Image

from src.datasets.dataset import VLEODataset


class COWCDataset(VLEODataset):
    url = "https://gdo152.llnl.gov/cowc/download/cowc-m-everything.txz"

    base_path = "./data/cowc-m/"
    processed_path = "outputs/DetectionPatches_512x512"
    count_file = "object_count.csv"
    count_header = ["Neg_Count", "Other_Count", "Pickup_Count", "Sedan_Count", "Unknown_Count"]
    annotation_header = ["Object_Class", "xmin", "ymin", "xmax", "ymax"]
    categories = {0: "negative", 1: "other", 2: "pickup", 3: "sedan", 4: "unknown"}

    def __init__(self):
        pass

    def construct_hf_dataset(self, config_name: str = None) -> Dataset:
        df_metadata = pd.read_csv(os.path.join(self.base_path, self.processed_path, self.count_file))
        df_metadata["image"] = df_metadata.apply(
            lambda x: os.path.join(self.base_path, self.processed_path, x["Folder_Name"], x["File_Name"]),
            axis=1
        )
        df_metadata["city"] = df_metadata["Folder_Name"].apply(lambda x: x.split("_", maxsplit=2)[0])
        df_metadata["source"] = df_metadata["Folder_Name"].apply(lambda x: x.split("_", maxsplit=2)[1])

        df_metadata = df_metadata[df_metadata[self.count_header].sum(axis=1) > 0]

        if config_name:
            df_metadata = df_metadata[df_metadata["city"] == config_name]

        df_metadata.reset_index(drop=True, inplace=True)
        assert len(df_metadata) > 0

        objects = []

        for image_file in df_metadata["image"]:
            annotation_file = image_file.replace(".jpg", ".txt")

            if not os.path.exists(annotation_file):
                objects.append({"bbox": None, "category": None})
                print("Non-existent annotation or no object", annotation_file)
                continue

            df_objects = pd.read_csv(annotation_file, delimiter=" ", names=self.annotation_header)
            objects.append({
                "bbox": df_objects[self.annotation_header[1:]].to_numpy().tolist(),
                "categories": df_objects[self.annotation_header[0]].astype(int).replace(self.categories).to_numpy().tolist()
            })

        df_metadata["objects"] = objects
        hf_dataset = Dataset.from_pandas(df_metadata)

        return hf_dataset


def main():
    dataset = COWCDataset()

    for city in ["Columbus", "Potsdam", "Selwyn", "Toronto", "Utah", "Vaihingen"]:
        hf_dataset = dataset.construct_hf_dataset(city).cast_column(column="image", feature=Image(decode=True))
        print(hf_dataset[0])

        hf_dataset.push_to_hub("danielz01/cowc-m", config_name=city)


if __name__ == "__main__":
    main()
