import base64
import re
import json
import os.path
import string
from glob import glob

import pandas as pd
from datasets import Dataset, Image
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from sklearn.metrics import classification_report

from src.datasets.dataset import VLEODataset
from src.utils.gpt4v_chat import encode_image


class RSICDDataset(VLEODataset):
    dir_name = "RSICD"
    image_dir = "RSICD_images"
    class_dir = "txtclasses_rsicd"
    caption_fname = "dataset_rsicd.json"
    caption_splits = ["train", "val"]

    CAPTION_KEY = "images"

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = os.path.join(data_dir, self.dir_name)
        self.caption_path = os.path.join(self.data_dir, self.caption_fname)
        with open(self.caption_path) as f:
            self.captions = json.load(f)[self.CAPTION_KEY]

        self.img2landcover = {}
        for land_cover_file in glob(os.path.join(data_dir, self.dir_name, self.class_dir, "*.txt")):
            land_cover_type = os.path.basename(land_cover_file).replace(".txt", "")
            land_cover_type = string.capwords(re.sub("([^-])([A-Z][a-z-]+)", r"\1 \2", land_cover_type))
            with open(land_cover_file) as f:
                for line in f.readlines():
                    img_file = line.strip()
                    self.img2landcover[img_file] = land_cover_type
        self.land_cover_types = sorted(list(set(self.img2landcover.values())))

    def convert_hf_dataset(self, split: str) -> "Dataset":
        assert split in self.caption_splits
        captions = list(filter(lambda x: x["split"] == split, self.captions))
        assert len(captions) > 0
        hf_dicts = []
        for caption in captions:
            dataset_entry = {
                "image": os.path.join(self.data_dir, self.image_dir, caption["filename"]),
                "path": caption["filename"],
                "img_id": caption["imgid"],
                "captions": [x["raw"] for x in caption["sentences"]],
                "caption_ids": [x["sentid"] for x in caption["sentences"]]
            }
            hf_dicts.append(dataset_entry)

        result_dataset = Dataset.from_pandas(pd.DataFrame(hf_dicts)).cast_column("image", Image())
        return result_dataset

    def convert_clipscore_captions(self):
        hf_dataset = self.convert_hf_dataset("val")
        clipscore_reference = {
            os.path.basename(path).removesuffix(".jpg"): captions for path, captions in
            zip(hf_dataset["path"], hf_dataset["captions"])
        }

        with open("./data/RSICD/reference_captions.json", "w") as dest:
            json.dump(clipscore_reference, dest, indent=4)

        return clipscore_reference

    def get_multiple_choice_prompts(self) -> str:
        prompt = "Which of the following choices best describes the given image? You should select only one choice. "
        for i, land_cover_type in enumerate(self.land_cover_types):
            prompt += f"{i}. {land_cover_type} "

        return prompt

    def get_captioning_prompts(self) -> str:
        prompt = ("Generate a caption for the image in one sentence. Make sure to follow the following instructions:\n"
                  "1. Describe all the important parts of the remote sensing image.\n2. Do not start the sentences "
                  "with “There is” when there are more than one object in an image.\n3. Do not use the vague concept "
                  "of words like large, tall, many, in the absence of contrast.\n4. Do not use direction nouns, "
                  "such as north, south, east and west.\n5. The sentences should contain at least six words.")

        return prompt

    def parse_openai(self):
        src = "./data/RSICD/gpt-4v-captioning.jsonl"
        df = pd.read_json(src, lines=True)
        df["caption"] = df["response"].apply(lambda x: x["choices"][0]["message"]["content"])

        # df["classification"] = df["response"].apply(lambda x: x[0].split(". ", maxsplit=2)[1])
        # df["caption"] = df["response"].apply(lambda x: x[1])
        # del df["response"]

        # df["label"] = df["img"].replace(self.img2landcover)

        # print(classification_report(y_pred=df["classification"], y_true=df["label"]))

        df.to_csv("./data/RSICD/gpt-4v.csv", index=False)

        clipscore_captions = {
            x.removesuffix(".jpg"): y for x, y in zip(df["img"], df["caption"])
        }

        with open("./data/RSICD/gpt-4v-captioning.json", "w") as dest:
            json.dump(clipscore_captions, dest, indent=4)

    def parse_qwen(self):
        df_caption = pd.read_json("./data/RSICD/qwen_rsicd_captions.jsonl", lines=True)
        clipscore_captions = {
            x.removesuffix(".jpg"): y for x, y in zip(df_caption["img"], df_caption["response"])
        }

        with open("./data/RSICD/qwen-captioning.json", "w") as dest:
            json.dump(clipscore_captions, dest, indent=4)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def query_hf(self, img_path: str):
        api_url = "https://api-inference.huggingface.co/models/microsoft/git-base"
        headers = {"Authorization": f"Bearer {os.environ['HF_API_KEY']}"}

        with open(img_path, "rb") as src:
            data = src.read()
            response = requests.post(api_url, headers=headers, data=data)
            print(response.json())
            if not response.ok:
                raise IOError(str(response.json()))

        return None, response.json()[0]["generated_text"]


def main():
    dataset = RSICDDataset()
    dataset.parse_qwen()
    dataset.parse_openai()
    dataset.convert_clipscore_captions()

    print(dataset.get_captioning_prompts())
    print(dataset.land_cover_types)
    hf_dataset = dataset.convert_hf_dataset("val")
    print(len(hf_dataset))

    result_path = "./data/RSICD/gpt-4v-captioning.jsonl"
    if os.path.exists(result_path):
        final_results = []
        with open(result_path) as f:
            for line in f.readlines():
                final_results.append(json.loads(line.strip()))
    else:
        final_results = []
    type_count = {x: 0 for x in dataset.land_cover_types}
    for data_item in hf_dataset:
        data_item.pop("image")
        land_cover_type = dataset.img2landcover[data_item["path"]]

        if any([x["img"] == data_item["path"] for x in final_results]):
            print(f"Skipping {data_item['path']}")
            type_count[land_cover_type] += 1
            continue
        if type_count[land_cover_type] > 100:
            continue

        fpath = os.path.join(dataset.data_dir, dataset.image_dir, data_item["path"])
        assert os.path.exists(fpath)
        query_payload, query_result = dataset.query_openai(
            encode_image(fpath), system=dataset.system_message, user=dataset.get_captioning_prompts()
        )
        print(data_item["path"], query_result["choices"][0]["message"]["content"])

        if query_result:
            final_results.append({"img": data_item["path"], "response": query_result})
            type_count[land_cover_type] += 1

        with open(result_path, "w") as f:
            for line in final_results:
                f.write(json.dumps(line) + "\n")

    # hf_dataset.push_to_hub("danielz01/rsicd", split=split_name)
    print(hf_dataset[0])


def validate():
    dataset = RSICDDataset()
    print(dataset.get_captioning_prompts())
    dataset.parse_qwen()


if __name__ == "__main__":
    main()
