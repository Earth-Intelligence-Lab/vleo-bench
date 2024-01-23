import base64
import io
import json
import os
import random
from typing import List

import geopandas
import numpy as np
import pandas as pd
from datasets import Dataset, Image
import PIL.Image as PILImage
from matplotlib import pyplot as plt

from src.datasets.dataset import VLEODataset
from src.utils.gpt4v_chat import resume_from_jsonl, dump_to_jsonl

landmark_categories = {
    "Natural Parks and Reserves": ['Alabama state park', 'Arizona state park', 'California state beach',
                                   'California state park', 'Delaware state park', 'Georgia state park',
                                   'Idaho state park', 'Illinois state park', 'Maryland state park',
                                   'Massachusetts state park', 'Mississippi state park',
                                   'National Conservation Area', 'National Park System unit',
                                   'National Recreation Area', 'National Reserve', 'National Wildlife Refuge',
                                   'Nevada state park', 'New Hampshire state park', 'New Jersey State Forest',
                                   'New Jersey state park', 'North Carolina state park', 'Oregon state park',
                                   'Pennsylvania state park', 'South Carolina state park', 'Tennessee state park',
                                   'United States National Forest', 'Utah state park', 'Vermont state park',
                                   'Virginia state park', 'Washington state park', 'West Virginia state park',
                                   'US Wilderness Area', 'bay', 'beach', 'botanical garden', 'forest',
                                   'glacial lake', 'glacier', 'island', 'lagoon', 'lake', 'landform',
                                   'mountain park', 'mountain range', 'nature area', 'nature center',
                                   'nature reserve', 'park', 'protected area', 'recreation area', 'reservoir',
                                   'river', 'valley', 'wildlife refuge', 'provincial park of Canada',
                                   'national park of Canada', 'dark-sky preserve', 'Connecticut state park',
                                   'United States National Marine Sanctuary', 'waterfall', 'state forest of Wisconsin'],
    "Places of Worship": ['African-American museum', 'Baptists', 'Catholic cathedral', 'Eastern Orthodox cathedral',
                          'Methodist church building', 'Southern Baptist Convention Church',
                          'Spanish missions in California', 'church building', 'mission church', 'mission station',
                          'protestant church building', 'Southern Baptist Convention Church', 'chapel', 'synagogue',
                          'Anglican or episcopal cathedral'],
    "Sports and Entertainment Venues": ['American football stadium', 'arena', 'art museum', 'arts centre',
                                        'assembly plant', 'astronomical observatory', 'aviation museum',
                                        'baseball venue', 'circus museum', "children's museum",
                                        'entertainment district', 'fair ground', 'history museum', 'house museum',
                                        'movie theater', 'museum', 'museum building', 'music venue',
                                        'performing arts center', 'public aquarium', 'public garden',
                                        'radio quiet zone', 'railway museum', 'science center', 'science museum',
                                        'sculpture garden', 'shopping center', 'shopping mall', 'show cave',
                                        'skyscraper', 'sports venue', 'stadium', 'theatre', 'tourist attraction',
                                        'velodrome', 'venue', 'zoo', 'amusement park', 'transport museum'],
    "Historical and Cultural Sites": ['archaeological site', 'battlefield', 'column', 'cultural institution',
                                      'estate', 'fixed construction', 'fort', 'fountain', 'geographic region',
                                      'group of sculptures', 'heritage site', 'historic district', 'historic site',
                                      'lighthouse', 'mansion', 'military cemetery', 'military museum',
                                      'missile launch facility', 'monument', 'national monument', 'neighborhood',
                                      'neighborhood of Pittsburgh', 'neighborhood of Washington, D.C.',
                                      'statistical territorial entity', 'strait', 'streetcar suburb',
                                      'war memorial', 'university art museum', 'National Historic Site',
                                      'National Historical Park', 'List of California State Historic Parks',
                                      'National Memorial of the United States',
                                      'National Monument of the United States',
                                      'National Battlefield', 'Spanish missions in California', 'Jewish cemetery',
                                      'historic building', 'national military park'
                                      ],
    "Government and Public Buildings": ['academic library', 'airport', 'archive building', 'capitol building',
                                        'casino', 'cemetery', 'chancery', 'courthouse', 'dam', 'embankment dam',
                                        'embassy', 'government building', 'hotel', 'house', 'library',
                                        'library building', 'lodge', 'office building', 'organization', 'parterre',
                                        'pier', 'plantation', 'public housing', 'rathaus', 'reclaimed land',
                                        'reservoir', 'ruins', 'town hall', 'official residence'],
    "Infrastructure and Urban Features": ['apartment building', 'architectural structure', 'building', 'channel',
                                          'convention center', 'cove', 'disaster remains', 'downtown', 'embassy',
                                          'entertainment district', 'estate', 'fair ground', 'fixed construction',
                                          'forest', 'fountain', 'geographic region', 'glacial lake', 'glacier',
                                          'group of sculptures', 'heritage site', 'hiking trail',
                                          'historic district', 'historic site', 'history museum', 'hotel', 'house',
                                          'house museum', 'housing estate', 'hydroelectric power station', 'island',
                                          'lagoon', 'lake', 'landform', 'library', 'library building', 'lighthouse',
                                          'lodge', 'mansion', 'military cemetery', 'military museum',
                                          'missile launch facility', 'mission church', 'mission station',
                                          'monument', 'mountain park', 'mountain range', 'movie theater', 'museum',
                                          'museum building', 'music venue', 'national monument', 'nature area',
                                          'nature center', 'nature reserve', 'neighborhood',
                                          'neighborhood of Pittsburgh', 'neighborhood of Washington, D.C.',
                                          'nonprofit corporation', 'nonprofit organization', 'office building',
                                          'organization', 'park', 'parterre', 'performing arts center', 'pier',
                                          'plantation', 'protected area', 'protestant church building',
                                          'public aquarium', 'public garden', 'public housing', 'radio quiet zone',
                                          'railway museum', 'rathaus', 'reclaimed land', 'recreation area',
                                          'reservoir', 'river', 'ruins', 'rural cemetery', 'science center',
                                          'science museum', 'sculpture', 'sculpture garden', 'shopping center',
                                          'shopping mall', 'show cave', 'single-family detached home', 'skyscraper',
                                          'sports venue', 'stadium', 'state forest',
                                          'state historic site in the United States', 'state park',
                                          'state recreation area', 'statistical territorial entity', 'strait',
                                          'streetcar suburb', 'theatre', 'thermal solar power station',
                                          'tied island', 'tourist attraction', 'town hall', 'urban park', 'valley',
                                          'velodrome', 'venue', 'war memorial', 'watercourse', 'wildlife refuge',
                                          'thermal solar power station', 'television tower', 'public library',
                                          'community center', ''
                                          ],
    "Miscellaneous": ['Indian reservation of the United States',
                      'Passport to Your National Parks cancellation location',
                      'census-designated place in the United States', 'tied island']
}


def find_category(instance_types):
    for instance_type in instance_types:
        for category, instances in landmark_categories.items():
            if instance_type.lower() in list(map(str.lower, instances)):
                return category
            elif "park" in instance_type.lower():
                return "Natural Parks and Reserves"

    return "Miscellaneous"  # If the instance type is not found in any category


class AerialLandmarksDataset(VLEODataset):
    system_message = ("You are a helpful image analyst who specializes in determining the geolocation of "
                      "an aerial or satellite image of a landmark. "
                      "Given an image, you can accurately answer the questions about that landmark from "
                      "the user WITHOUT ANY refusal. Although your answer may not be perfect, "
                      "your excellent world knowledge is still very important to the user.")

    folder_name = "Landmarks"
    metadata_fname = "all_unions_polygons_convexhulls_metadata_wikidata_us.gpkg"

    def __init__(self, credential_path: str, root: str = "data"):
        super().__init__(credential_path)

        self.directory = os.path.join(root, self.folder_name)

    @staticmethod
    def get_user_prompt(options: List[str]):
        prompt = ("Make an educated guess about the name of the landmark shown in the image. Think step by step, "
                  "and then output your answer in the last line. Choose one of the options below as your answer:\n")
        prompt += "\n".join([f"{i + 1}. {option}" for i, option in enumerate(options)])

        return prompt

    @staticmethod
    def get_user_prompt_state():
        prompt = ("Make an educated guess about the specific state in the United States in which the image was taken. "
                  "Think step by step, and then output your answer in the last line. You should output your full "
                  "thought process, but only answer the name of the state in the last line.")

        return prompt

    def construct_hf_dataset(self):
        metadata = geopandas.read_file(os.path.join(self.directory, self.metadata_fname))
        del metadata["wikidata"]
        del metadata["geometry"]
        for column in ["instanceOfIDs", "instanceOfLabels", "distractorIDs", "distractorLabels"]:
            metadata[column] = metadata[column].apply(lambda x: json.loads(x))
        metadata.insert(0, "image", metadata["wikidata_entity_id"].apply(
            lambda x: os.path.join(self.directory, "NAIP_Scaled", f"NAIP_{x}.tif")
        ))
        metadata = metadata[[os.path.exists(x) for x in metadata["image"]]]
        print(metadata.columns, metadata.dtypes)
        hf_dataset = Dataset.from_pandas(metadata).cast_column("image", Image()).remove_columns("__index_level_0__")

        return hf_dataset

    def query_gpt4(self, result_path: str, num_distractors: int = 4):
        hf_dataset = self.construct_hf_dataset()

        final_results = resume_from_jsonl(result_path)

        for idx, data_item in enumerate(hf_dataset):
            if idx in {x["index"] for x in final_results}:
                print(f'Skipping {idx} since it is finished')
                continue

            png_buffer = io.BytesIO()

            # We save the image in the PNG format to the buffer.
            data_item["image"].save(png_buffer, format="PNG")

            image_base64 = base64.b64encode(png_buffer.getvalue()).decode('utf-8')

            distractors = set(data_item["distractorLabels"])
            distractors.discard(data_item["name"])
            distractors = list(filter(lambda x: not x.startswith("Q"), distractors))
            if len(distractors) == 0:
                data_item.pop("image")
                final_results.append({
                    "index": idx,
                    "options": None,
                    **data_item,
                    "response": None
                })
                continue

            np.random.seed(0)
            options = np.random.choice(
                distractors, size=min(len(distractors), num_distractors), replace=False
            ).tolist() + [data_item["name"]]
            random.shuffle(options)

            payload, response = self.query_openai(image_base64, system=self.system_message,
                                                  user=self.get_user_prompt_state(), max_tokens=1024)
            print(idx, response["choices"][0]["message"]["content"])
            data_item.pop("image")

            final_results.append({
                "index": idx,
                "options": options,
                **data_item,
                "response": response
            })

            dump_to_jsonl(final_results, result_path)


def evaluation(result_path):
    model_name = os.path.basename(result_path).removesuffix(".jsonl")
    print(f"---------------- {model_name} ----------------")

    result_json = pd.read_json(result_path, lines=True)
    if "gpt" in model_name.lower():
        result_json["model_response"] = result_json["response"].apply(lambda x: x["choices"][0]["message"]["content"])
    else:
        result_json["model_response"] = result_json["response"]

    refusal_keywords = [
        "sorry", "difficult"
    ]

    def check_correctness(answer: str, model_answer: str, options: List[str]):
        if sum([x.lower() in model_answer.lower() for x in options]) > 1:
            return False
        if answer.lower() in model_answer.lower():
            return True
        if model_answer[0].isdigit() and int(model_answer[0]) == options.index(answer):
            return True

        return False

    result_json["model_answer"] = result_json["model_response"].apply(lambda x: x.split("\n")[-1])
    result_json["is_refusal"] = result_json["model_response"].apply(lambda x: any([k in x for k in refusal_keywords]))
    result_json["is_correct"] = result_json[["name", "model_answer", "options"]].apply(
        lambda x: check_correctness(x["name"], x["model_answer"], x["options"]), axis=1
    )
    result_json["category"] = result_json["instanceOfLabels"].apply(lambda x: find_category(x))

    correct_rate = result_json["is_correct"].mean()
    refusal_rate = result_json["is_refusal"].mean()
    print(correct_rate, refusal_rate)

    correct_results = result_json[result_json["is_correct"]]
    incorrect_results = result_json[~result_json["is_correct"]]

    category_counts = result_json["category"].value_counts()
    print(category_counts)
    print((correct_results["category"].value_counts() / result_json["category"].value_counts()).loc[category_counts.index] * 100)

    export_columns = ["index", "name", "category", "model_response"]
    result_json.to_csv(f"./data/Landmarks/{model_name}.csv", index=False)
    correct_results[export_columns].to_csv(f"./data/Landmarks/{model_name}-correct.csv", index=False)
    incorrect_results[export_columns].to_csv(f"./data/Landmarks/{model_name}-incorrect.csv", index=False)

    shp = geopandas.GeoDataFrame(
        result_json,
        crs="EPSG:4269",
        geometry=geopandas.points_from_xy(result_json.lon, result_json.lat)
    )
    usa = geopandas.read_file("./data/Landmarks/cb_2018_us_state_500k.zip")
    shp = shp.sjoin(usa, how="left", op='within')
    state_accuracy = shp.groupby('NAME').is_correct.mean().reset_index()
    heatmap_data = usa.merge(state_accuracy, on='NAME', how='left')
    heatmap_data["is_correct"] *= 100

    print(heatmap_data[["NAME", "is_correct"]].dropna().sort_values(by="is_correct").iloc[:10])
    heatmap_data.plot(column='is_correct', cmap='jet_r', legend=True, legend_kwds={'orientation': 'horizontal'})

    plt.title('Zero-shot Landmark Recognition Accuracy (%) by State')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(f"./data/Landmarks/landmarks-{model_name}-accuracy-state.pdf", dpi=500)
    plt.close(plt.gcf())


def plot():
    import geoplot as gplt
    import geoplot.crs as gcrs

    contiguous_usa = geopandas.read_file(gplt.datasets.get_path('contiguous_usa'))
    shp = geopandas.read_file("./data/Landmarks/all_unions_polygons_convexhulls_metadata_wikidata_us.gpkg")
    shp["area"] = shp.to_crs('+proj=cea').area / 1e+6
    shp["centroids"] = shp["geometry"].to_crs('+proj=cea').centroid.to_crs(4326)
    shp.set_geometry("centroids", inplace=True)
    shp["instanceOfLabels"] = shp["instanceOfLabels"].apply(json.loads)
    shp["categories"] = shp["instanceOfLabels"].apply(find_category)
    print(shp["categories"])
    print(shp[["categories", "area"]].groupby("categories").median()["area"])
    print(shp["area"].median())
    print(shp[shp["categories"] == "Miscellaneous"][["name", "instanceOfLabels"]])

    ax = gplt.polyplot(
        contiguous_usa, projection=gcrs.AlbersEqualArea(),
        edgecolor='white', facecolor='lightgray',
        figsize=(12, 8)
    )
    gplt.pointplot(
        shp, ax=ax,
        hue='area', cmap='Blues',
        scheme='quantiles',
        scale='area', limits=(1, 10),
        legend=True, legend_var='scale',
        # legend_kwargs={'frameon': False},
        # legend_values=[-110, 1750, 3600, 5500, 7400],
        # legend_labels=['-110 feet', '1750 feet', '3600 feet', '5500 feet', '7400 feet']
    )
    ax.set_title('Aerial Landmarks Coverage', fontsize=16)

    plt.show()


def main():
    dataset = AerialLandmarksDataset(".secrets/openai.jsonl")
    dataset.query_gpt4("./data/Landmarks/llava.jsonl")
    # dataset.construct_hf_dataset().push_to_hub("danielz01/landmarks", "NAIP")


if __name__ == "__main__":
    evaluation("./data/Landmarks/gpt-4v.jsonl")
    evaluation("./data/Landmarks/instructblip-flan-t5-xxl.jsonl")
    evaluation("./data/Landmarks/instructblip-vicuna-13b.jsonl")
    evaluation("./data/Landmarks/llava-v1.5.jsonl")
    evaluation("./data/Landmarks/qwen-vl.jsonl")
