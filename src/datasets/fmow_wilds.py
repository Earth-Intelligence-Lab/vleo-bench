import os.path
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, ClassLabel
from src.datasets.dataset import VLEODataset


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

    def __init__(self, root: str = "./data", split: str = "test", download: str = False):
        assert split in self.splits

        self.root = root
        self.split = split

        self.categories_normalized = {x: " ".join([y.capitalize() for y in x.split("_")]) for x in self.categories}
        self.directory = os.path.join(root, self.directory_name)
        self.image_directory = os.path.join(self.directory, self.image_directory_name)

        self.metadata = pd.read_csv(os.path.join(self.directory, self.metadata_name))
        self.metadata = self.metadata[self.metadata["split"] == self.split]
        self.metadata.reset_index(drop=True, inplace=True)

        from wilds import get_dataset
        from wilds.datasets.fmow_dataset import FMoWDataset

        self.wilds_dataset: FMoWDataset = get_dataset(
            dataset="fmow", root_dir=self.root, download=download
        )
        self.wilds_subset = self.wilds_dataset.get_subset(self.split)
        self.subset_metadata = self.wilds_dataset.metadata.loc[self.wilds_subset.indices]

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


def main():
    for split in ["test", "id_test", "val", "id_val", "train"]:
        dataset = FMoWWILDSDataset(root=".", split=split)
        hf_dataset = dataset.convert_to_hf_dataset()
        os.makedirs(f"hf/{split}", exist_ok=True)
        hf_dataset.save_to_disk(f"hf/{split}", num_proc=32)
        hf_dataset.push_to_hub("danielz01/fMoW", config_name="WILDS", split=split)


if __name__ == "__main__":
    main()
