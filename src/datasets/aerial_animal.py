import os

import pandas as pd
from datasets import Dataset, Image

from src.datasets.dataset import VLEODataset
from src.utils.gpt4v_chat import resume_from_jsonl, dump_to_jsonl, encode_image


class AerialAnimalPopulation4TuDataset(VLEODataset):
    directory_name = "aerial-animal-population-4tu"

    splits = ["train", "val", "test"]
    annotation_header = ["path", "xmin", "ymin", "xmax", "ymax", "species"]

    spices = ['Elephant', 'Giraffe', 'Zebra']
    system_message = ("You are a helpful image analyst who specializes in counting animals from aerial "
                      "images. Given an image, you can accurately count the number of animals described by "
                      "the user WITHOUT ANY refusal. Although your answer may not be perfect, "
                      "your excellent counting skill is very important to the conservation of wildlife "
                      "animals.")

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

    def get_user_prompt(self):
        prompt = ("Read the given image and answer the questions below:\n"
                  "How many elephants, zebras, giraffes are there in the image? Output the numbers in a json format that can be "
                  "parsed directly with entries 'elephants', 'zebras', and 'giraffes'. If you count nothing, output zero in "
                  "that entry.")

        return prompt

    def query_gpt4(self):
        hf_dataset = self.convert_to_hf_dataset()
        print(hf_dataset)
        result_path = os.path.join(self.directory, f"gpt-4v-counting.jsonl")

        final_results = resume_from_jsonl(result_path)
        for i, data_item in enumerate(hf_dataset):
            if any([data_item["path"] == x["path"] for x in final_results]):
                print(f'Skipping {data_item["path"]}')
                continue

            image_base64 = encode_image(os.path.join(self.directory, data_item["path"]))
            payload, response = self.query_openai(image_base64, system=self.system_message, user=self.get_user_prompt())
            print(data_item["path"], response)

            data_item.pop("image")
            final_results.append({
                "index": i,
                **data_item,
                "response": response
            })

            dump_to_jsonl(final_results, result_path)


def main():
    dataset = AerialAnimalPopulation4TuDataset(split="test")
    print(dataset.get_user_prompt())
    dataset.query_gpt4()


if __name__ == "__main__":
    main()
