import os
from glob import glob
from typing import Optional

import numpy as np
import pandas as pd
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
        "beach": "Beach", "bridge": "Bridge", "cemetery": "Cemetery", "chaparral": "Chaparral",
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


def evaluation(result_path: str):
    model_name = os.path.basename(result_path).removesuffix(".jsonl")
    result_dir = os.path.dirname(result_path)
    print(f"---------------- {model_name} ----------------")

    cm_path = os.path.join(result_dir, f"{model_name}.pdf")
    csv_path = os.path.join(result_dir, f"{model_name}.csv")
    refusal_path = os.path.join(result_dir, f"{model_name}-refusal.csv")
    incorrect_path = os.path.join(result_dir, f"{model_name}-incorrect.csv")

    result_json = pd.read_json(result_path, lines=True)

    if "gpt" in model_name.lower():
        result_json["model_response"] = result_json["response"].apply(lambda x: x["choices"][0]["message"]["content"])
    else:
        result_json["model_response"] = result_json["response"]

    categories_normalized = sorted(set(PatternNetDataset.categories_reassigned.values()))

    def parse_response(response: str):
        for category in categories_normalized:
            if category.lower() in response.strip().lower():
                return category
        return "Refused"

    refusal_keywords = [
        "sorry", "difficult"
    ]

    result_json["label"] = result_json["label"].apply(lambda x: PatternNetDataset.categories_reassigned[x.lower().replace(" ", "_")])
    result_json["model_answer"] = result_json["model_response"].apply(parse_response)
    result_json["is_refusal"] = result_json["model_response"].apply(lambda x: any([k in x for k in refusal_keywords]))
    result_json["is_refusal"] = np.logical_and(result_json["is_refusal"], result_json["model_answer"] != "Refused")
    result_json["is_correct"] = result_json["model_answer"] == result_json["label"]

    rr = result_json["is_refusal"].mean()
    acc = result_json["is_correct"].mean()

    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    print(result_json[["model_response", "model_answer", "label"]])

    print(f"RR {rr:.4f} | Accuracy {acc:.4f}")
    print(classification_report(y_true=result_json["label"], y_pred=result_json["model_answer"]))
    clf_report = pd.DataFrame(
        classification_report(y_true=result_json["label"], y_pred=result_json["model_answer"], output_dict=True)
    ).transpose()

    SMALL_SIZE = 4
    MEDIUM_SIZE = 5
    BIGGER_SIZE = 6

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, ax = plt.subplots(figsize=(20, 20))
    ConfusionMatrixDisplay.from_predictions(y_true=result_json["label"], y_pred=result_json["model_answer"], ax=ax).plot()
    plt.xticks(rotation=90)
    # plt.tight_layout()
    plt.savefig(cm_path, bbox_inches='tight', dpi=500)
    plt.savefig(cm_path.replace(".pdf", ".png"), bbox_inches='tight', dpi=500)

    clf_report = clf_report.round(decimals=2)
    clf_report.to_csv(os.path.join(result_dir, f"{model_name}-classification.csv"))

    del result_json["response"]

    result_incorrect = result_json[~result_json["is_correct"]]
    result_refusal = result_json[result_json["is_refusal"]]

    result_json.to_csv(csv_path, index=False)
    result_incorrect.to_csv(incorrect_path, index=False)
    result_refusal.to_csv(refusal_path, index=False)


if __name__ == "__main__":
    evaluation("./data/PatternNet/gpt-4v-reassigned.jsonl")
    evaluation("./data/PatternNet/train-instructblip-flan-t5-xxl.jsonl")
    evaluation("./data/PatternNet/train-instructblip-vicuna-13b.jsonl")
    evaluation("./data/PatternNet/train-llava-v1.5-13b.jsonl")
    evaluation("./data/PatternNet/train-Qwen-VL-Chat.jsonl")
