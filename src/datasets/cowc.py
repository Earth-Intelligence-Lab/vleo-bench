import os
from typing import List, Optional

import numpy as np
import pandas as pd
from datasets import Dataset, Image
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

from src.datasets.dataset import VLEODataset
from src.utils.counting import parse_digit_response, calculate_counting_metrics, plot_scatter
from src.utils.gpt4v_chat import encode_pil_image, resume_from_jsonl, dump_to_jsonl, encode_image


class COWCDataset(VLEODataset):
    url = "https://gdo152.llnl.gov/cowc/download/cowc-m-everything.txz"

    base_path = "./data/cowc-m/"
    processed_path = "outputs/DetectionPatches_512x512"
    count_file = "object_count.csv"
    count_header = ["Neg_Count", "Other_Count", "Pickup_Count", "Sedan_Count", "Unknown_Count"]
    annotation_header = ["Object_Class", "xmin", "ymin", "xmax", "ymax"]
    categories = {0: "negative", 1: "other", 2: "pickup", 3: "sedan", 4: "unknown"}

    # system_message = ("You are a helpful image analyst who specializes in counting objects and pinpointing the "
    #                   "geolocation from satellite and aerial images given natural language instruction. "
    #                   "Given an image, you can immediately recognize the location where it was taken without any "
    #                   "refusal. You always truthfully answer a user's questions. If you are not sure about something, "
    #                   "DO NOT answer false information.")

    def __init__(self, credential_path: str):
        super().__init__(credential_path)

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
                "categories": df_objects[self.annotation_header[0]].astype(int).replace(
                    self.categories).to_numpy().tolist()
            })

        df_metadata["objects"] = objects
        hf_dataset = Dataset.from_pandas(df_metadata).cast_column(column="image", feature=Image(decode=True))

        return hf_dataset

    @staticmethod
    def get_user_prompt(target_cities: Optional[List[str]]):
        if target_cities:
            prompt = ("Read the given image and answer the following questions:\n1. How many cars are there in this "
                      "image? You only need to answer a number.\n2. In a new line, answer which of the following "
                      "locations the image is taken from. Choose the best option: ")
            prompt += "; ".join([f"{i + 1}. {city}" for i, city in enumerate(target_cities)])
        else:
            prompt = "How many cars are there in this image? You only need to answer a number."

        return prompt

    def query_gpt4(self, target_cities: Optional[List[str]] = None, max_queries: int = 1000):
        if target_cities is None:
            target_cities = ["Potsdam", "Selwyn", "Toronto", "Utah"]
        count_per_class = max_queries // len(target_cities)

        for city in target_cities:
            result_path = os.path.join(self.base_path, f"gpt-4v-answers-{city}.jsonl")
            print(f"Saving to {result_path}")
            final_results = resume_from_jsonl(result_path)

            hf_dataset = self.construct_hf_dataset(config_name=city)
            count = 0
            for i, data_item in enumerate(hf_dataset):
                if any([data_item["File_Name"] == x["File_Name"] for x in final_results]):
                    print(f'Skipping {data_item["File_Name"]}')
                    count += 1
                    continue

                if count > count_per_class:
                    break

                image_base64 = encode_image(
                    os.path.join(self.base_path, self.processed_path, data_item["Folder_Name"], data_item["File_Name"])
                )
                user_prompt = self.get_user_prompt(None)

                payload, response = self.query_openai(image_base64, system=self.system_message, user=user_prompt)
                print(data_item["File_Name"], response["choices"][0]["message"]["content"])

                data_item.pop("image")
                final_results.append({
                    "city": city,
                    "index": i,
                    **data_item,
                    "response": response
                })

                dump_to_jsonl(final_results, result_path)

                count += 1


def main():
    dataset = COWCDataset(credential_path="./.secrets/openai.json")

    for city in ["Columbus", "Potsdam", "Selwyn", "Toronto", "Utah", "Vaihingen"]:
        hf_dataset = dataset.construct_hf_dataset(city).cast_column(column="image", feature=Image(decode=True))
        print(hf_dataset[0])


def test():
    dataset = COWCDataset(credential_path="./.secrets/openai.json")
    dataset.query_gpt4()


def evaluation(result_path, ax=None):
    *model, city = os.path.basename(result_path).removesuffix(".jsonl").split("-")
    model_name = "-".join(model)
    print(model_name, city)

    result_json = pd.read_json(result_path, lines=True)
    if "gpt" in model_name:
        result_json["model_response"] = result_json["response"].apply(lambda x: x["choices"][0]["message"]["content"])
    else:
        result_json["model_response"] = result_json["response"]
    if "objects" not in result_json:
        result_json["objects"] = pd.read_json(
            os.path.join(os.path.dirname(result_path), f"gpt-4v-{city}.jsonl"), lines=True
        )["objects"]

    result_json["parsed_response"] = result_json["model_response"].apply(parse_digit_response)
    result_json["count"] = result_json["objects"].apply(lambda x: len(x["bbox"]))
    result_json_no_refusal = result_json[result_json["parsed_response"] != -1].copy()
    result_json_no_refusal["Predicted Count"] = result_json_no_refusal["parsed_response"]
    result_json_no_refusal["True Count"] = result_json_no_refusal["count"]

    rr, (mape, mape_no_refusal), (mae, mae_no_refusal), (r2, r2_no_refusal) = calculate_counting_metrics(result_json, result_json_no_refusal)

    # print(f"MAPE & MAPE (No Refusal) & R2 & R2 (No Refusal) & Refusal Rate")
    # print(f"{mape:.3f} & {mape_no_refusal:.3f} & {r2:.3f} & {r2_no_refusal:.3f} & {rr:.3f}")

    print(f"MAE (No Refusal) & MAPE (No Refusal) & R2 (No Refusal) & Refusal Rate")
    print(f"{mae_no_refusal:.3f} & {mape_no_refusal:.3f} & {r2_no_refusal:.3f} & {rr:.2f}")

    plot_scatter(result_json_no_refusal, ax=ax)
    if ax:
        ax.set_title(model_name)

    plt.savefig(result_path.replace(".jsonl", ".pdf"))
    plt.savefig(result_path.replace(".jsonl", ".png"))


def combine_cities():
    from glob import glob

    model_names = {
        "-".join(os.path.basename(x).removesuffix(".jsonl").split("-")[:-1]) for x in glob("./data/cowc-m/*.jsonl")
    }
    cities = ["Utah", "Potsdam", "Toronto", "Selwyn"]
    for model_name in model_names:
        dfs = []
        for city in cities:
            print(f"./data/cowc-m/{model_name}-{city}.jsonl")
            df = pd.read_json(f"./data/cowc-m/{model_name}-{city}.jsonl", lines=True)
            dfs.append(df)
        dfs = pd.concat(dfs)
        dfs.to_json(f"./data/cowc-m/{model_name}-combined.jsonl", orient="records", lines=True)
        print(f"./data/cowc-m/{model_name}-combined.jsonl")


if __name__ == "__main__":
    from glob import glob

    # combine_cities()
    for result_file in sorted(glob("./data/cowc-m/*-combined.jsonl")):
        evaluation(result_file)
