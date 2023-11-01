import os
from glob import glob
import xml.etree.ElementTree as ET

from datasets import Dataset, Image

from src.datasets.dataset import VLEODataset


class ForestDamageDataset(VLEODataset):
    url = "https://storage.googleapis.com/public-datasets-lila/larch-casebearer/Data_Set_Larch_Casebearer.zip"

    base_path = "./data/ForestDamages/"

    def __init__(self):
        pass

    def xml2dict(self, xml_file) -> dict:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size_info, *_ = root.findall("size")
        img_info = {
            "width": int(size_info.find("width").text),
            "height": int(size_info.find("height").text),
            "depth": int(size_info.find("depth").text)
        }

        objects = {"bbox": [], "categories": [], "damage": []}
        objects_meta = {"truncated": [], "pose": [], "difficult": []}

        for obj in root.findall('object'):
            # Extract bounding box coordinates and category
            bbox = obj.find('bndbox')
            # Assuming one bbox per object
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            objects["bbox"].append([xmin, ymin, xmax, ymax])

            tree_category = obj.find('tree')
            tree_damage = obj.find('damage')
            objects["categories"].append(tree_category.text if tree_category is not None else None)
            objects["damage"].append(tree_damage.text if tree_damage is not None else None)

            for k in objects_meta.keys():
                key_text = obj.find(k).text
                if key_text.isdigit():
                    key_text = int(key_text)
                objects_meta[k].append(key_text)

        return {
            **img_info,
            "objects": objects,
            "objects_meta": objects_meta
        }

    def construct_hf_dataset(self) -> Dataset:
        metadata = []  # {"image": [], "objects": [], "count": [], "path": []}

        image_files = glob(os.path.join(self.base_path, "**", "*.JPG"), recursive=True)
        for image_file in image_files:
            annotation_file = image_file.replace("Images", "Annotations").replace(".JPG", ".xml")
            image_file_metadata = {
                "image": image_file,
                "path": image_file
            }
            if not os.path.exists(annotation_file):
                metadata.append(image_file_metadata)
                print(annotation_file)
                continue

            image_file_metadata.update(self.xml2dict(annotation_file))
            metadata.append(image_file_metadata)

        hf_dataset = Dataset.from_list(metadata)

        return hf_dataset


def main():
    dataset = ForestDamageDataset()
    hf_dataset = dataset.construct_hf_dataset().cast_column(column="image", feature=Image(decode=True))
    # annotated_dataset = hf_dataset.filter(lambda example: example["objects"])

    print(hf_dataset[1000])
    # print(annotated_dataset)

    hf_dataset.push_to_hub("danielz01/forest-damage")


if __name__ == "__main__":
    main()
