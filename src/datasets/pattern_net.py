import os
from glob import glob

import numpy as np
from datasets import Dataset, Image

from src.datasets.dataset import VLEODataset
from src.utils.gpt4v_chat import resume_from_jsonl, dump_to_jsonl, encode_pil_image, encode_image


class PatternNetDataset(VLEODataset):
    """PatternNet dataset.

        The `PatternNet <https://sites.google.com/view/zhouwx/dataset>`__
        dataset is a dataset for remote sensing scene classification and image retrieval.

        Dataset features:

        * 30,400 images with 6-50 cm per pixel resolution (256x256 px)
        * three spectral bands - RGB
        * 38 scene classes, 800 images per class

        Dataset format:

        * images are three-channel jpgs

        Dataset classes:

        0. airplane
        1. baseball_field
        2. basketball_court
        3. beach
        4. bridge
        5. cemetery
        6. chaparral
        7. christmas_tree_farm
        8. closed_road
        9. coastal_mansion
        10. crosswalk
        11. dense_residential
        12. ferry_terminal
        13. football_field
        14. forest
        15. freeway
        16. golf_course
        17. harbor
        18. intersection
        19. mobile_home_park
        20. nursing_home
        21. oil_gas_field
        22. oil_well
        23. overpass
        24. parking_lot
        25. parking_space
        26. railway
        27. river
        28. runway
        29. runway_marking
        30. shipping_yard
        31. solar_panel
        32. sparse_residential
        33. storage_tank
        34. swimming_pool
        35. tennis_court
        36. transformer_station
        37. wastewater_treatment_plant

        If you use this dataset in your research, please cite the following paper:

        * https://doi.org/10.1016/j.isprsjprs.2018.01.004
        """

    url = "https://drive.google.com/file/d/127lxXYqzO6Bd0yZhvEbgIfz95HaEnr9K"
    md5 = "96d54b3224c5350a98d55d5a7e6984ad"
    filename = "PatternNet.zip"
    directory = os.path.join("PatternNet", "images")

    categories = [
        'airplane',
        'baseball_field',
        'basketball_court',
        'beach',
        'bridge',
        'cemetery',
        'chaparral',
        'christmas_tree_farm',
        'closed_road',
        'coastal_mansion',
        'crosswalk',
        'dense_residential',
        'ferry_terminal',
        'football_field',
        'forest',
        'freeway',
        'golf_course',
        'harbor',
        'intersection',
        'mobile_home_park',
        'nursing_home',
        'oil_gas_field',
        'oil_well',
        'overpass',
        'parking_lot',
        'parking_space',
        'railway',
        'river',
        'runway',
        'runway_marking',
        'shipping_yard',
        'solar_panel',
        'sparse_residential',
        'storage_tank',
        'swimming_pool',
        'tennis_court',
        'transformer_station',
        'wastewater_treatment_plant'
    ]

    categories_reassigned = {
        "airplane": "Airplane", "baseball_field": "Baseball Field", "basketball_court": "Basketball Court",
        "beach": "Beach", "Bridge": "Bridge", "cemetery": "Cemetery", "chaparral": "Chaparral",
        "christmas_tree_farm": "Christmas Tree Farm", "closed_road": "Closed Road", "coastal_mansion": "Coastal Mansion",
        "crosswalk": "Crosswalk", "dense_residential": "Residential", "ferry_terminal": "Harbor",
        "football_field": "Football Field", "forest": "Forest", "freeway": "Freeway", "golf_course": "Golf Course",
        "harbor": "Harbor", "intersection": "Intersection", "mobile_home_park": "Mobile Home Park",
        "nursing_home": "Nursing Home", "oil_gas_field": "Oil Gas Field", "oil_well": "Oil Well",
        "overpass": "Overpass", "parking_lot": "Parking Space", "parking_space": "Parking Space", "railway": "Railway",
        "river": "River", "runway": "Runway", "runway_marking": "Runway", "shipping_yard": "Shipping Yard",
        "solar_panel": "Solar Panel", "sparse_residential": "Residential", "storage_tank": "Storage Tank",
        "swimming_pool": "Swimming Pool", "tennis_court": "Tennis Court", "transformer_station": "Transformer Station",
        "wastewater_treatment_plant": "Wastewater Treatment Plant"
    }

    def __init__(self, credential_path: str, root: str = "data", download: bool = False,
                 checksum: bool = False) -> None:
        """Initialize a new PatternNet dataset instance.

        Args:
            root: root directory where dataset can be found
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        super().__init__(credential_path)
        self.root = root
        self.download = download
        self.checksum = checksum

        self.folders = sorted(glob(os.path.join(root, self.directory, "*")))
        self.folder2class = {
            folder: " ".join([os.path.basename(x).capitalize() for x in folder.split("_")]) for folder in self.folders
        }

        self.categories_normalized = [" ".join([y.capitalize() for y in x.split("_")]) for x in self.categories]

    def construct_hf_dataset(self):
        hf_dict = {"image": [], "label": [], "path": []}
        for folder in self.folders:
            folder_img = sorted(glob(os.path.join(folder, "*.jpg")))
            img_paths = [os.path.relpath(x, os.path.join(self.root, self.directory)) for x in folder_img]
            img_labels = [self.folder2class[folder]] * len(folder_img)

            hf_dict["image"].extend(folder_img)
            hf_dict["path"].extend(img_paths)
            hf_dict["label"].extend(img_labels)

        return Dataset.from_dict(hf_dict)

    def get_user_prompt(self):
        prompt = ("You are given a satellite image and a list of land usage types or object names. Classify the image "
                  "into one of the following options. Choose the best option that describes the given image. A list "
                  "of possible options:\n")
        prompt += ";\n".join([f"{i + 1}. {option}" for i, option in enumerate(sorted(set(self.categories_reassigned.values())))])
        prompt += "\nYour choice of one option that best describes the image:"

        return prompt

    def query_gpt4(self, result_path: str, max_queries: int = 1000):
        hf_dataset = self.construct_hf_dataset()
        np.random.seed(0)
        selected_indices = set(np.random.choice(range(len(hf_dataset)), size=max_queries, replace=False).tolist())
        type_count = {x: 0 for x in self.categories_normalized}
        print(type_count)
        type_threshold = 26  # max_queries // len(self.categories)

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

            image_base64 = encode_image(data_item["image"])
            payload, response = self.query_openai(image_base64, system=self.system_message, user=self.get_user_prompt())
            print(idx, response["choices"][0]["message"]["content"])
            data_item.pop("image")
            final_results.append({
                "index": idx,
                **data_item,
                "response": response
            })

            type_count[image_category] += 1
            dump_to_jsonl(final_results, result_path)


def main():
    dataset = PatternNetDataset(credential_path=".secrets/openai.jsonl")
    print(dataset.folders, dataset.folder2class)
    dataset.query_gpt4("./data/PatternNet/gpt-4v-reassigned.jsonl", max_queries=1500)


if __name__ == "__main__":
    main()
