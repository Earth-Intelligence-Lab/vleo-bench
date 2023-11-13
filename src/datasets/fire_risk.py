import os
from glob import glob
from datasets import Dataset, Image


class FireRisk:
    directory = "FireRisk"
    splits = ["train", "val"]
    classes = [
        "High",
        "Low",
        "Moderate",
        "Non-burnable",
        "Very_High",
        "Very_Low",
        "Water",
    ]

    def __init__(self, root: str = "./data", split: str = "val"):
        assert split in self.splits

        self.root = root
        self.split = split

    def construct_hf_dataset(self) -> Dataset:
        hf_dict = {"image": [], "label": [], "path": []}
        for class_name in self.classes:
            folder_img = sorted(glob(os.path.join(self.root, self.directory, self.split, class_name, "*.png")))
            assert len(folder_img) > 0
            img_paths = [os.path.relpath(x, os.path.join(self.root, self.directory, self.split)) for x in folder_img]
            img_labels = [class_name.replace("_", " ")] * len(folder_img)

            hf_dict["image"].extend(folder_img)
            hf_dict["label"].extend(img_labels)
            hf_dict["path"].extend(img_paths)

        hf_dataset = Dataset.from_dict(hf_dict).cast_column("image", Image())

        return hf_dataset


def main():
    for split in ["train", "val"]:
        dataset = FireRisk(split=split)
        hf_dataset = dataset.construct_hf_dataset()
        print(hf_dataset)


if __name__ == "__main__":
    main()
