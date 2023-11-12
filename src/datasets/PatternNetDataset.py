import os
from glob import glob

from datasets import Dataset, Image


class PatternNetDataset:
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

    def __init__(self, root: str = "data", download: bool = False, checksum: bool = False) -> None:
        """Initialize a new PatternNet dataset instance.

        Args:
            root: root directory where dataset can be found
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        self.root = root
        self.download = download
        self.checksum = checksum

        self.folders = sorted(glob(os.path.join(root, self.directory, "*")))
        self.folder2class = {
            folder: " ".join([os.path.basename(x).capitalize() for x in folder.split("_")]) for folder in self.folders
        }

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


def main():
    dataset = PatternNetDataset()
    print(dataset.folders, dataset.folder2class)
    hf_dataset = dataset.construct_hf_dataset().cast_column("image", Image())


if __name__ == "__main__":
    main()
