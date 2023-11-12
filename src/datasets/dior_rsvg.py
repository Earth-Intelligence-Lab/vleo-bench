import os.path
import xml.etree.ElementTree as ET
import numpy as np
from datasets import Dataset, Image


def get_file_list(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name, files in os.walk(root) for f in files if
            f.endswith(file_type)]


class DIORRSVGDataset:
    folder_name = "DIOR-RSVG"
    jpeg_directory = os.path.join(folder_name, "JPEGImages")
    annotation_directory = os.path.join(folder_name, "Annotations")
    category_name = {
        'basketballcourt': 'Basketball Court', 'overpass': 'Overpass', 'dam': 'Dam',
        'groundtrackfield': 'Ground Track Field', 'storagetank': 'Storage Tank', 'stadium': 'Stadium',
        'airplane': 'Airplane', 'golffield': 'Golf Field', 'baseballfield': 'Baseball Field',
        'Expressway-Service-area': 'Expressway Service Area', 'bridge': 'Bridge', 'tenniscourt': 'Tennis Court',
        'harbor': 'Harbor', 'windmill': 'Windmill', 'ship': 'Ship', 'chimney': 'Chimney', 'airport': 'Airport',
        'Expressway-toll-station': 'Expressway Toll Station', 'trainstation': 'Train Station', 'vehicle': 'Vehicle'
    }

    def __init__(self, root: str = "./data", split: str = "test"):
        assert split in ["train", "test", "val"]

        self.root = root
        self.split = split

        self.object_list = self._get_objects()
        self.categories = sorted(list({x["object"]["categories"] for x in self.object_list}))

    def _get_objects(self):
        count = 0
        object_list = []
        annotations = get_file_list(os.path.join(self.root, self.annotation_directory), ".xml")
        with open(os.path.join(self.root, self.folder_name, f"{self.split}.txt"), "r") as f:
            split_index = [int(x.strip()) for x in f.readlines()]
        for annotation_path in annotations:
            root = ET.parse(annotation_path).getroot()
            for member in root.findall('object'):
                if count in split_index:
                    img_file = os.path.join(self.root, self.jpeg_directory, root.find("./filename").text)
                    bbox = np.array([
                        int(member[2][0].text), int(member[2][1].text),
                        int(member[2][2].text), int(member[2][3].text)
                    ], dtype=np.int32)
                    class_name = member[0].text
                    text = member[3].text

                    object_list.append({
                        "image": img_file,
                        "object": {
                            "bbox": bbox, "categories": class_name,
                            "categories_normalized": self.category_name[class_name], "captions": text}
                    })
                count += 1

        return object_list

    def _get_images(self):
        images = {x["image"] for x in self.object_list}
        image_dict = {
            x: {
                "path": os.path.basename(x),
                "objects": {"bbox": [], "categories": [], "categories_normalized": [], "captions": []}
            } for x in images
        }
        for bbox_object in self.object_list:
            for k, v in bbox_object["object"].items():
                image_dict[bbox_object["image"]]["objects"][k].append(v)
        image_list = [{"image": k, **v} for k, v in image_dict.items()]

        return image_list

    def construct_hf_dataset(self) -> Dataset:
        return Dataset.from_list(self._get_images())


def main():
    for split in ["train", "val", "test"]:
        dataset = DIORRSVGDataset(split=split)
        hf_dataset = dataset.construct_hf_dataset().cast_column("image", Image())


if __name__ == "__main__":
    main()
