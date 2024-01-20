import json
import os

import pandas as pd
from datasets import Dataset, Image

from src.datasets.dataset import VLEODataset
from src.utils.counting import parse_digit_response, calculate_counting_metrics, plot_scatter
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

    def __init__(self, credential_path: str, root: str = "./data", split: str = "test"):
        super().__init__(credential_path)
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


def evaluation(result_path, ax=None):
    result_json = pd.read_json(result_path, lines=True)
    if "gpt" in result_path:
        result_json["response"] = result_json["response"].apply(lambda x: x["choices"][0]["message"]["content"])
    else:
        result_json["response"] = result_json["response"]
    if "objects" not in result_json:
        result_json["objects"] = pd.read_json(
            os.path.join(os.path.dirname(result_path), f"gpt-4v-counting.jsonl"), lines=True
        )["objects"]

    result_json["model_response"] = result_json["response"].apply(
        lambda x: x.replace("json\n", "").replace("```", "").replace("\n", "")
    )

    def parse_response(x):
        try:
            ret = json.loads(x)
        except json.decoder.JSONDecodeError:
            ret = ""
        return ret
    result_json["parsed_response"] = result_json["model_response"].apply(parse_response).apply(
        lambda x: sum(x.values()) if isinstance(x, dict) else 0
    )

    result_json["count"] = result_json["objects"].apply(lambda x: len(x["bbox"]))
    result_json_no_refusal = result_json[result_json["parsed_response"] != ""].copy()

    _, (mape, mape_no_refusal), (mae, mae_no_refusal), (r2, r2_no_refusal) = calculate_counting_metrics(
        result_json, result_json_no_refusal
    )

    rr = 1 - len(result_json_no_refusal) / len(result_json)

    print(f"MAPE & MAPE (No Refusal) & R2 & R2 (No Refusal) & Refusal Rate")
    print(f"{mape:.3f} & {mape_no_refusal:.3f} & {r2:.3f} & {r2_no_refusal:.3f} & {rr:.3f}")

    *model, _ = os.path.basename(result_path).removesuffix(".jsonl").split("-")
    model_name = "-".join(model)
    plot_scatter(result_json_no_refusal, ax=ax)
    if ax:
        ax.set_title(model_name)


if __name__ == "__main__":
    for file in [
        "./data/aerial-animal-population-4tu/gpt-4v-counting.jsonl",
        "./data/aerial-animal-population-4tu/Qwen-VL-Chat-counting.jsonl",
        "./data/aerial-animal-population-4tu/llava-v1.5-13b-counting.jsonl",
    ]:
        evaluation(file)

