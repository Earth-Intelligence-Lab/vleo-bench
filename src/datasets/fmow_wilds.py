import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, ClassLabel, load_dataset

from src.datasets.dataset import VLEODataset
from src.utils.gpt4v_chat import resume_from_jsonl, encode_pil_image, dump_to_jsonl


class FMoWWILDSDataset(VLEODataset):
    directory_name = "fMoW-WILDS"
    image_directory_name = "images"
    metadata_name = "rgb_metadata.csv"
    splits = ["test", "id_test", "val", "id_val", "train"]

    categories = [
        "airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture",
        "archaeological_site", "barn", "border_checkpoint", "burial_site", "car_dealership",
        "construction_site", "crop_field", "dam", "debris_or_rubble", "educational_institution",
        "electric_substation", "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
        "gas_station", "golf_course", "ground_transportation_station", "helipad", "hospital",
        "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", "military_facility",
        "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
        "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track",
        "railway_bridge", "recreational_facility", "road_bridge", "runway", "shipyard", "shopping_mall",
        "single-unit_residential", "smokestack", "solar_farm", "space_facility", "stadium", "storage_tank",
        "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening", "waste_disposal",
        "water_treatment_facility", "wind_farm", "zoo"
    ]

    def __init__(self, credential_path: str, root: str = "./data", split: str = "test", download: str = False):
        super().__init__(credential_path)

        assert split in self.splits

        self.root = root
        self.split = split

        self.categories_normalized = [" ".join([y.capitalize() for y in x.split("_")]) for x in self.categories]

        from wilds import get_dataset
        from wilds.datasets.fmow_dataset import FMoWDataset

        # self.wilds_dataset: FMoWDataset = get_dataset(
        #     dataset="fmow", root_dir=self.root, download=download
        # )
        # self.wilds_subset = self.wilds_dataset.get_subset(self.split)
        # self.subset_metadata = self.wilds_dataset.metadata.loc[self.wilds_subset.indices]

    def _hf_item_generator(self):
        for i in tqdm(range(len(self.wilds_subset))):
            x, y, metadata = self.wilds_subset[i]
            yield {
                "image": x,
                "label": y,
                "domain_labels": metadata,
                "domain_labels_readable": {
                    x: self.wilds_subset.metadata_map.get(x, {metadata[i].item(): None})[metadata[i].item()] for i, x in
                    enumerate(self.wilds_subset.metadata_fields)},
                **self.subset_metadata.iloc[i].to_dict()
            }

    def convert_to_hf_dataset(self) -> Dataset:
        return (Dataset.
                from_generator(self._hf_item_generator).
                cast_column('label', ClassLabel(names=self.categories)))

    def get_user_prompt(self):
        prompt = ("You are given a satellite image and a list of land usage types. Choose one land use type "
                  "that best describes the image. A list of possible land use types:\n")
        prompt += ";\n".join([f"{i + 1}. {option}" for i, option in enumerate(self.categories_normalized)])
        prompt += "\nYour choice of one land use type that best describes the image:"

        return prompt

    def query_gpt4(self, result_path: str, max_queries: int = 1000):
        hf_dataset = load_dataset("danielz01/fMoW", "WILDS", split=self.split)
        np.random.seed(0)
        selected_indices = set(np.random.choice(range(len(hf_dataset)), size=max_queries, replace=False).tolist())
        type_count = {x: 0 for x in range(len(self.categories))}
        type_threshold = 16  # max_queries // len(self.categories)

        final_results = resume_from_jsonl(result_path)
        for result in final_results:
            type_count[result["label"]] += 1
        for idx in selected_indices:
            data_item = hf_dataset[int(idx)]
            image_category = data_item["label"]

            if any([idx == x["index"] for x in final_results]):
                print(f'Skipping {data_item} since it is finished')
                continue
            if type_count[image_category] >= type_threshold:
                print(f'Skipping {data_item} due to class balancing')
                continue

            image_base64 = encode_pil_image(data_item["image"])
            payload, response = self.query_openai(image_base64, system=self.system_message, user=self.get_user_prompt())
            print(idx, response)
            data_item.pop("image")
            final_results.append({
                "index": idx,
                **data_item,
                "response": response
            })

            type_count[image_category] += 1
            dump_to_jsonl(final_results, result_path)


def main():
    for split in ["test", "id_test", "val", "id_val", "train"]:
        dataset = FMoWWILDSDataset(credential_path=".secrets/openai.jsonl", root=".", split=split)
        hf_dataset = dataset.convert_to_hf_dataset()
        os.makedirs(f"hf/{split}", exist_ok=True)
        hf_dataset.save_to_disk(f"hf/{split}", num_proc=32)
        hf_dataset.push_to_hub("danielz01/fMoW", config_name="WILDS", split=split)


def evaluation():
    for split in ["id_test"]:
        dataset = FMoWWILDSDataset(credential_path=".secrets/openai.jsonl", root=".", split=split)
        dataset.query_gpt4(f"/home/danielz/PycharmProjects/vleo-bench/data/fMoW/{split}-gpt4-v.jsonl", max_queries=11327)


if __name__ == "__main__":
    evaluation()
