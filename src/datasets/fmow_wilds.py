import os.path

import pandas as pd
from datasets import Dataset

from src.datasets.dataset import VLEODataset


class FMoWWILDSDataset(VLEODataset):
    directory_name = "fMoW-WILDS"
    image_directory_name = "images"
    metadata_name = "rgb_metadata.csv"
    splits = ["train", "val", "test", "seq"]

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

    def __init__(self, root: str = "./data", split: str = "test"):
        assert split in self.splits

        self.root = root
        self.split = split

        self.categories_normalized = {x: " ".join([y.capitalize() for y in x.split("_")]) for x in self.categories}
        self.directory = os.path.join(root, self.directory_name)
        self.image_directory = os.path.join(self.directory, self.image_directory_name)

        self.metadata = pd.read_csv(os.path.join(self.directory, self.metadata_name))
        self.metadata = self.metadata[self.metadata["split"] == self.split]
        self.metadata.reset_index(drop=True, inplace=True)

    def convert_to_hf_dataset(self) -> Dataset:
        hf_list = {}
