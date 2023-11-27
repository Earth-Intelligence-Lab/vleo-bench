import base64
import io
import json
import os
import random
from typing import List

import geopandas
import numpy as np
from datasets import Dataset, Image
import PIL.Image as PILImage
from src.datasets.dataset import VLEODataset
from src.utils.gpt4v_chat import resume_from_jsonl, dump_to_jsonl


class AerialLandmarksDataset(VLEODataset):
    system_message = ("You are a helpful image analyst who specializes in determining the geolocation of "
                      "an aerial or satellite image of a landmark. "
                      "Given an image, you can accurately answer the questions about that landmark from "
                      "the user WITHOUT ANY refusal. Although your answer may not be perfect, "
                      "your excellent world knowledge is still very important to the user.")

    folder_name = "Landmarks"
    metadata_fname = "all_unions_polygons_convexhulls_metadata_wikidata_us.gpkg"

    def __init__(self, credential_path: str, root: str = "data"):
        super().__init__(credential_path)

        self.directory = os.path.join(root, self.folder_name)

    @staticmethod
    def get_user_prompt(options: List[str]):
        prompt = ("Make an educated guess about the name of the landmark shown in the image. Think step by step, "
                  "and then output your answer in the last line. Choose one of the options below as your answer:\n")
        prompt += "\n".join([f"{i + 1}. {option}" for i, option in enumerate(options)])

        return prompt

    def construct_hf_dataset(self):
        metadata = geopandas.read_file(os.path.join(self.directory, self.metadata_fname))
        del metadata["wikidata"]
        del metadata["geometry"]
        for column in ["instanceOfIDs", "instanceOfLabels", "distractorIDs", "distractorLabels"]:
            metadata[column] = metadata[column].apply(lambda x: json.loads(x))
        metadata.insert(0, "image", metadata["wikidata_entity_id"].apply(
            lambda x: os.path.join(self.directory, "NAIP_Scaled", f"NAIP_{x}.tif")
        ))
        metadata = metadata[[os.path.exists(x) for x in metadata["image"]]]
        print(metadata.columns, metadata.dtypes)
        hf_dataset = Dataset.from_pandas(metadata).cast_column("image", Image()).remove_columns("__index_level_0__")

        return hf_dataset

    def query_gpt4(self, result_path: str, num_distractors: int = 4):
        hf_dataset = self.construct_hf_dataset()

        final_results = resume_from_jsonl(result_path)

        for idx, data_item in enumerate(hf_dataset):
            if idx in {x["index"] for x in final_results}:
                print(f'Skipping {idx} since it is finished')
                continue

            png_buffer = io.BytesIO()

            # We save the image in the PNG format to the buffer.
            data_item["image"].save(png_buffer, format="PNG")

            image_base64 = base64.b64encode(png_buffer.getvalue()).decode('utf-8')

            distractors = set(data_item["distractorLabels"])
            distractors.discard(data_item["name"])
            distractors = list(filter(lambda x: not x.startswith("Q"), distractors))
            if len(distractors) == 0:
                data_item.pop("image")
                final_results.append({
                    "index": idx,
                    "options": None,
                    **data_item,
                    "response": None
                })
                continue

            np.random.seed(0)
            options = np.random.choice(
                distractors, size=min(len(distractors), num_distractors), replace=False
            ).tolist() + [data_item["name"]]
            random.shuffle(options)

            payload, response = self.query_openai(image_base64, system=self.system_message,
                                                  user=self.get_user_prompt(options), max_tokens=1024)
            print(idx, response["choices"][0]["message"]["content"])
            data_item.pop("image")

            final_results.append({
                "index": idx,
                "options": options,
                **data_item,
                "response": response
            })

            dump_to_jsonl(final_results, result_path)


def main():
    dataset = AerialLandmarksDataset(".secrets/openai.jsonl")
    # dataset.query_gpt4("/home/danielz/PycharmProjects/vleo-bench/data/Landmarks/gpt-4v.jsonl")
    # dataset.construct_hf_dataset().push_to_hub("danielz01/landmarks", "NAIP")


if __name__ == "__main__":
    main()
