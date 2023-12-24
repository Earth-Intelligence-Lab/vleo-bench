import json
import os
from glob import glob
from typing import Union, Dict, List

import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image as PILImage
from datasets import Dataset, load_dataset
from matplotlib import pyplot as plt
from numpy._typing import NDArray
from rasterio.enums import Resampling
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
from torch import Tensor
from tqdm import tqdm

from src.datasets.dataset import VLEODataset
from src.utils.gpt4v_chat import resume_from_jsonl, dump_to_jsonl, encode_pil_image


def sort_sentinel2_bands(x: str) -> str:
    """Sort Sentinel-2 band files in the correct order."""
    x = os.path.basename(x).split("_")[-1]
    x = os.path.splitext(x)[0]
    if x == "B8A":
        x = "B08A"
    return x


class BigEarthNetDataset(VLEODataset):
    class_sets = {
        19: [
            "Urban fabric",
            "Industrial or commercial units",
            "Arable land",
            "Permanent crops",
            "Pastures",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland and sparsely vegetated areas",
            "Moors, heathland and sclerophyllous vegetation",
            "Transitional woodland, shrub",
            "Beaches, dunes, sands",
            "Inland wetlands",
            "Coastal wetlands",
            "Inland waters",
            "Marine waters",
        ],
        43: [
            "Continuous urban fabric",
            "Discontinuous urban fabric",
            "Industrial or commercial units",
            "Road and rail networks and associated land",
            "Port areas",
            "Airports",
            "Mineral extraction sites",
            "Dump sites",
            "Construction sites",
            "Green urban areas",
            "Sport and leisure facilities",
            "Non-irrigated arable land",
            "Permanently irrigated land",
            "Rice fields",
            "Vineyards",
            "Fruit trees and berry plantations",
            "Olive groves",
            "Pastures",
            "Annual crops associated with permanent crops",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland",
            "Moors and heathland",
            "Sclerophyllous vegetation",
            "Transitional woodland/shrub",
            "Beaches, dunes, sands",
            "Bare rock",
            "Sparsely vegetated areas",
            "Burnt areas",
            "Inland marshes",
            "Peatbogs",
            "Salt marshes",
            "Salines",
            "Intertidal flats",
            "Water courses",
            "Water bodies",
            "Coastal lagoons",
            "Estuaries",
            "Sea and ocean",
        ],
    }

    label_converter = {
        0: 0,
        1: 0,
        2: 1,
        11: 2,
        12: 2,
        13: 2,
        14: 3,
        15: 3,
        16: 3,
        18: 3,
        17: 4,
        19: 5,
        20: 6,
        21: 7,
        22: 8,
        23: 9,
        24: 10,
        25: 11,
        31: 11,
        26: 12,
        27: 12,
        28: 13,
        29: 14,
        33: 15,
        34: 15,
        35: 16,
        36: 16,
        38: 17,
        39: 17,
        40: 18,
        41: 18,
        42: 18,
    }

    splits_metadata = {
        "train": {
            "url": "https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/master/splits/train.csv"
                   "?inline=false",
            # noqa: E501
            "filename": "bigearthnet-train.csv",
            "md5": "623e501b38ab7b12fe44f0083c00986d",
        },
        "val": {
            "url": "https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/master/splits/val.csv?inline"
                   "=false",
            # noqa: E501
            "filename": "bigearthnet-val.csv",
            "md5": "22efe8ed9cbd71fa10742ff7df2b7978",
        },
        "test": {
            "url": "https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/master/splits/test.csv"
                   "?inline=false",
            # noqa: E501
            "filename": "bigearthnet-test.csv",
            "md5": "697fb90677e30571b9ac7699b7e5b432",
        },
    }
    metadata = {
        "s1": {
            "url": "https://bigearth.net/downloads/BigEarthNet-S1-v1.0.tar.gz",
            "md5": "94ced73440dea8c7b9645ee738c5a172",
            "filename": "BigEarthNet-S1-v1.0.tar.gz",
            "directory": "BigEarthNet-S1-v1.0",
        },
        "s2": {
            "url": "https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz",
            "md5": "5a64e9ce38deb036a435a7b59494924c",
            "filename": "BigEarthNet-S2-v1.0.tar.gz",
            "directory": "BigEarthNet-v1.0",
        },
    }
    cloud_snow_metadata = {
        "s2": ["patches_with_seasonal_snow.csv", "patches_with_cloud_and_shadow.csv"],
        "s2-rgb": ["patches_with_seasonal_snow.csv", "patches_with_cloud_and_shadow.csv"]
    }
    image_size = (120, 120)

    def __init__(self, credential_path: str, root: str = "data", split: str = "test", bands: str = "s2-rgb",
                 num_classes: int = 19) -> None:
        """Initialize a new BigEarthNet dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            bands: load Sentinel-1 bands, Sentinel-2, or both. one of {s1, s2, s2-rgb, all}
            num_classes: number of classes to load in target. one of {19, 43}

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        super().__init__(credential_path)
        assert split in self.splits_metadata
        assert bands in ["s1", "s2", "s2-rgb", "all"]
        assert num_classes in [43, 19]
        self.root = root
        self.split = split
        self.bands = bands
        self.num_classes = num_classes
        self.class2idx = {c: i for i, c in enumerate(self.class_sets[43])}
        self.folders = self._load_folders()

    def _load_folders(self) -> list[dict[str, str]]:
        """Load folder paths.

        Returns:
            list of dicts of s1 and s2 folder paths
        """
        filename = self.splits_metadata[self.split]["filename"]
        dir_s1 = self.metadata["s1"]["directory"]
        dir_s2 = self.metadata["s2"]["directory"]

        with open(os.path.join(self.root, filename)) as f:
            lines = f.read().strip().splitlines()
            pairs = [line.split(",") for line in lines]

        folders = [
            {
                "s1": os.path.join(self.root, dir_s1, pair[1]),
                "s2": os.path.join(self.root, dir_s2, pair[0]),
            }
            for pair in pairs
        ]
        return folders

    def _load_paths(self, index: int) -> list[str]:
        """Load paths to band files.

        Args:
            index: index to return

        Returns:
            list of file paths
        """
        if self.bands == "all":
            folder_s1 = self.folders[index]["s1"]
            folder_s2 = self.folders[index]["s2"]
            paths_s1 = glob(os.path.join(folder_s1, "*.tif"))
            paths_s2 = glob(os.path.join(folder_s2, "*.tif"))
            paths_s1 = sorted(paths_s1)
            paths_s2 = sorted(paths_s2, key=sort_sentinel2_bands)
            paths = paths_s1 + paths_s2
        elif self.bands == "s1":
            folder = self.folders[index]["s1"]
            paths = glob(os.path.join(folder, "*.tif"))
            paths = sorted(paths)
        elif self.bands == "s2-rgb":
            folder = self.folders[index]["s2"]
            b02_path = glob(os.path.join(folder, "*_B02.tif"))
            assert len(b02_path) == 1
            b02_path, *_ = b02_path
            paths = [b02_path.replace("B02", band_name) for band_name in ["B04", "B03", "B02"]]
        else:
            folder = self.folders[index]["s2"]
            paths = glob(os.path.join(folder, "*.tif"))
            paths = sorted(paths, key=sort_sentinel2_bands)

        return paths

    def _load_meta(self, index: int) -> Dict:
        if "s2" in self.bands:
            folder = self.folders[index]["s2"]
        else:
            folder = self.folders[index]["s1"]

        path = glob(os.path.join(folder, "*.json"))[0]
        with open(path) as f:
            meta_data = json.load(f)

        return meta_data

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target label
        """
        labels = self._load_meta(index)["labels"]

        # labels -> indices
        indices = [self.class2idx[label] for label in labels]

        # Map 43 to 19 class labels
        if self.num_classes == 19:
            indices_optional = [self.label_converter.get(idx) for idx in indices]
            indices = [idx for idx in indices_optional if idx is not None]

        target = torch.zeros(self.num_classes, dtype=torch.long)
        target[indices] = 1
        return target

    def _load_image(self, index: int) -> Union[NDArray, Tensor]:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        paths = self._load_paths(index)
        images = []
        for path in paths:
            # Bands are of different spatial resolutions
            # Resample to (120, 120)
            with rasterio.open(path) as dataset:
                array = dataset.read(
                    indexes=1,
                    out_shape=self.image_size,
                    out_dtype="int32",
                    resampling=Resampling.bilinear,
                )
                images.append(array)
        arrays: "NDArray[np.int_]" = np.stack(images, axis=0)
        # tensor = torch.from_numpy(arrays).float()
        return arrays

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.folders)

    def _hf_item_generator(self):
        for i in tqdm(range(len(self))):
            img = img_as_ubyte(rescale_intensity(self._load_image(i))).transpose(1, 2, 0)
            img = PILImage.fromarray(img)
            yield {
                "img": img,
                **self._load_meta(i)
            }

    def construct_hf_dataset(self, ) -> Dataset:
        return Dataset.from_generator(self._hf_item_generator)

    def get_user_prompt(self):
        prompt = ("You are given a satellite image and a list of land cover types. Choose all the land cover types "
                  "shown up in the image. A list of possible land cover types:\n")
        prompt += ";\n".join([f"{i + 1}. {option}" for i, option in enumerate(self.class_sets[19])])
        prompt += "\nOutput the all applicable options line by line, without any comment or further explanation."

        return prompt

    def query_gpt4(self, result_path: str, max_queries: int = 1000):
        hf_dataset = load_dataset("danielz01/BigEarthNet-S2-v1.0", split="test")

        np.random.seed(0)
        selected_indices = [x["index"] for x in
                            resume_from_jsonl("./data/BigEarthNet/gpt4-v.jsonl")]
        final_results = resume_from_jsonl(result_path)
        for idx in selected_indices:
            data_item = hf_dataset[int(idx)]
            if any([idx == x["index"] for x in final_results]):
                print(f'Skipping {data_item}')
                continue

            image_base64 = encode_pil_image(data_item["img"])
            payload, response = self.query_openai(image_base64, system=self.system_message, user=self.get_user_prompt())
            print(idx, response["choices"][0]["message"]["content"])
            data_item.pop("img")
            final_results.append({
                "index": idx,
                **data_item,
                "response": response
            })

            dump_to_jsonl(final_results, result_path)


def main():
    dataset = BigEarthNetDataset(credential_path=".secrets/openai_1701858440.jsonl", root=".", num_classes=19)
    dataset.query_gpt4("./data/BigEarthNet/gpt4-v-lines.jsonl")


def plot(confusion_matrices, label_names, save_path):
    fig, axes = plt.subplots(4, 4, figsize=(25, 25))
    axes = axes.ravel()
    for i, (ax, cm, label_name) in enumerate(zip(axes, confusion_matrices, label_names)):
        if label_name == "Land principally occupied by agriculture, with significant areas of natural vegetation":
            label_name = "Land principally occupied by agriculture,\nwith significant areas of natural vegetation"

        disp = ConfusionMatrixDisplay(confusion_matrices[i], display_labels=["Negative", "Positive"])
        disp.plot(ax=axes[i], values_format='.4g')

        disp.ax_.set_title(label_name)
        if i < 12:
            disp.ax_.set_xlabel('')
        if i % 4 != 0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()

    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.10, hspace=0.1)
    fig.colorbar(disp.im_, ax=axes)
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"))


def evaluation(result_path: str):
    model_name = os.path.basename(result_path).removesuffix(".jsonl")
    result_dir = os.path.dirname(result_path)
    print(f"---------------- {model_name} ----------------")

    cm_path = os.path.join(result_dir, f"{model_name}.pdf")
    csv_path = os.path.join(result_dir, f"{model_name}.csv")
    refusal_path = os.path.join(result_dir, f"{model_name}-refusal.csv")

    result_json = pd.read_json(result_path, lines=True)

    if "gpt" in model_name.lower():
        result_json["model_response"] = result_json["response"].apply(lambda x: x["choices"][0]["message"]["content"])
    else:
        result_json["model_response"] = result_json["response"]

    class2idx = {c: i for i, c in enumerate(BigEarthNetDataset.class_sets[43])}

    # Map 43 to 19 class labels
    def label_reassign(labels: List[str]):
        class_indices = [class2idx[label] for label in labels]
        indices_optional = [BigEarthNetDataset.label_converter.get(idx) for idx in class_indices]
        indices = [idx for idx in indices_optional if idx is not None]
        labels_reassigned = [BigEarthNetDataset.class_sets[19][x] for x in indices]

        return labels_reassigned

    # 43 to 19
    # label_converter = {x: BigEarthNetDataset.class_sets[19][BigEarthNetDataset.label_converter.get(i, x)] for i, x in
    #                    enumerate(BigEarthNetDataset.class_sets[43])}

    def parse_response(response: str):
        parsed_answers = []
        for category in sorted(BigEarthNetDataset.class_sets[19]):
            if category.lower() in response.strip().lower():
                parsed_answers.append(category)
        return parsed_answers if parsed_answers else ["Refused"]

    refusal_keywords = [
        "sorry", "difficult"
    ]

    result_json["labels"] = result_json["labels"].apply(label_reassign)
    result_json["model_answer"] = result_json["model_response"].apply(parse_response)
    result_json["is_refusal"] = result_json["model_response"].apply(lambda x: any([k in x for k in refusal_keywords]))
    result_json["is_refusal"] = np.logical_and(result_json["is_refusal"], result_json["model_answer"] != "Refused")
    # result_json["is_correct"] = result_json["model_answer"] == result_json["label"]

    rr = result_json["is_refusal"].mean()
    # acc = result_json["is_correct"].mean()

    from sklearn.metrics import classification_report, multilabel_confusion_matrix
    import matplotlib.pyplot as plt

    # print(result_json[["model_response", "model_answer", "labels"]])

    label_transformer = MultiLabelBinarizer()
    label_transformer.fit(y=result_json["labels"])

    y_pred = label_transformer.transform(result_json["model_answer"])
    y_true = label_transformer.transform(result_json["labels"])

    print(f"RR {rr:.4f}")
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=label_transformer.classes_))
    clf_report = pd.DataFrame(
        classification_report(y_true=y_true, y_pred=y_pred, output_dict=True, target_names=label_transformer.classes_)
    ).transpose()

    plot(
        multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred),
        label_transformer.classes_,
        cm_path
    )

    clf_report = clf_report.round(decimals=2)
    clf_report.to_csv(os.path.join(result_dir, f"{model_name}-classification.csv"))

    del result_json["response"]

    # result_incorrect = result_json[~result_json["is_correct"]]
    result_refusal = result_json[result_json["is_refusal"]]

    result_json.to_csv(csv_path, index=False)
    # result_incorrect.to_csv(incorrect_path, index=False)
    result_refusal.to_csv(refusal_path, index=False)


if __name__ == "__main__":
    evaluation("./data/BigEarthNet/gpt4-v-lines.jsonl")
    evaluation("./data/BigEarthNet/test-instructblip-flan-t5-xxl.jsonl")
    evaluation("./data/BigEarthNet/test-instructblip-vicuna-13b.jsonl")
    evaluation("./data/BigEarthNet/test-llava-v1.5-13b.jsonl")
    evaluation("./data/BigEarthNet/test-Qwen-VL-Chat.jsonl")
