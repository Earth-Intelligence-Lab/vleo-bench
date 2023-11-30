import os
from typing import List, Optional

import numpy as np
import pandas as pd
from datasets import Dataset, Image
from sklearn.metrics import mean_absolute_percentage_error

from src.datasets.dataset import VLEODataset
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
    dataset = COWCDataset()

    for city in ["Columbus", "Potsdam", "Selwyn", "Toronto", "Utah", "Vaihingen"]:
        hf_dataset = dataset.construct_hf_dataset(city).cast_column(column="image", feature=Image(decode=True))
        print(hf_dataset[0])


def test():
    dataset = COWCDataset()
    dataset.query_gpt4()


def evaluation(result_path):
    result_json = pd.read_json(result_path, lines=True)
    result_json["model_response"] = result_json["response"].apply(lambda x: x["choices"][0]["message"]["content"])

    def parse_response(x):
        try:
            ret = int(x)
        except ValueError:
            ret = -1
        return ret

    result_json["parsed_response"] = result_json["model_response"].apply(parse_response)
    result_json["count"] = result_json["objects"].apply(lambda x: len(x["bbox"]))
    result_json_no_refusal = result_json[result_json["parsed_response"] != -1]

    rr = (result_json["parsed_response"] == -1).mean()

    mape = mean_absolute_percentage_error(
        y_true=result_json["count"],
        y_pred=result_json["parsed_response"].replace(-1, 0)
    )
    mape_no_refusal = mean_absolute_percentage_error(
        y_true=result_json_no_refusal["count"],
        y_pred=result_json_no_refusal["parsed_response"]
    )

    r2 = np.corrcoef(result_json["count"], result_json["parsed_response"].replace(-1, 0))[0, 1] ** 2
    r2_no_refusal = np.corrcoef(
        result_json_no_refusal["count"], result_json_no_refusal["parsed_response"].replace(-1, 0)
    )[0, 1] ** 2

    print(os.path.basename(result_path))
    print(f"Refusal: {rr:.4f} \t | MAPE: {mape:.4f} | MAPE (No Refusal): {mape_no_refusal:.4f} | R2: {r2:.4f} | R2 ("
          f"No Refusal): {r2_no_refusal:.4f}")


if __name__ == "__main__":
    test()
