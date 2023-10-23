import geopandas
import pandas as pd

from src.datasets.dataset import VLEODataset


class IDTreesDataset(VLEODataset):
    train_url = "https://zenodo.org/records/3934932/files/IDTREES_competition_train_v2.zip?download=1"
    test_url = "https://zenodo.org/records/3934932/files/IDTREES_competition_test_v2.zip?download=1"

    train_labels = "./data/IDTReeS/train/Field/train_data.csv"
    train_shps = [
        "./data/IDTReeS/train/ITC/train_MLBS.shp",
        "./data/IDTReeS/train/ITC/train_OSBS.shp"
    ]
    crown2rs = "./data/IDTReeS/train/Field/itc_rsFile.csv"
    join_key = "indvdID"

    def __init__(self):
        pass

    def preprocess(self):
        train_labels = pd.read_csv(self.train_labels)
        train_rs_files = pd.read_csv(self.crown2rs)
        train_labels = pd.merge(
            left=train_labels,
            right=train_rs_files.drop("id", axis=1),
            on=self.join_key, how="left"
        )

        train_shps = pd.concat([geopandas.read_file(filename) for filename in self.train_shps])
        train_shps_meta = geopandas.GeoDataFrame(
            pd.merge(left=train_shps, right=train_labels, on=self.join_key, how="left")
        )
        train_shps_meta.to_file(self.train_labels.replace(".csv", ".gpkg"), driver="GPKG")


if __name__ == "__main__":
    dataset = IDTreesDataset()
    dataset.preprocess()
