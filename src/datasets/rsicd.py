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


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


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

    def get_multiple_choice_prompts(self) -> str:
        prompt = "Which of the following choices best describes the given image? You should select only one choice. "
        for i, land_cover_type in enumerate(self.land_cover_types):
            prompt += f"{i}. {land_cover_type} "
        prompt += "In a new line, generate a caption for the image in one sentence."

        return prompt

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def query_openai(self, base64_image: str):
        from openai import OpenAI

        client = OpenAI(organization="org-mJsdbmssTKFvmXiepg7eb1nX")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            # "OpenAI-Organization": "org-mJsdbmssTKFvmXiepg7eb1nX"
            # "OpenAI-Organization": "org-nAKVOfoTWrtlg0B9M5UaIkHV",
            "OpenAI-Organization": "org-VxKw6HqJ2zOiRv5AIMnVPoNJ"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful image analyst that specializes in satellite and aerial images. "
                                    "You always truthfully answer the user's question. If you are not sure about "
                                    "something, don't answer false information."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.get_multiple_choice_prompts()
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature": 0.2
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if not response.ok:
            raise IOError(str(response.json()))

        return payload, response.json()

    def parse_openai(self):
        src = "/home/danielz/PycharmProjects/vleo-bench/data/RSICD/gpt-4v.jsonl"
        df = pd.read_json(src, lines=True)
        df["response"] = df["response"].apply(lambda x: x["choices"][0]["message"]["content"].split("\n\n"))

        df["classification"] = df["response"].apply(lambda x: x[0].split(". ", maxsplit=2)[1])
        df["caption"] = df["response"].apply(lambda x: x[1])
        # del df["response"]

        df["label"] = df["img"].replace(self.img2landcover)

        print(classification_report(y_pred=df["classification"], y_true=df["label"]))

        df.to_csv("/home/danielz/PycharmProjects/vleo-bench/data/RSICD/gpt-4v.csv", index=False)

    def parse_qwen(self):
        src = "/home/danielz/PycharmProjects/vleo-bench/data/RSICD/qwen_rsicd_classification.jsonl"
        df = pd.read_json(src, lines=True)
        df_caption = pd.read_json("/home/danielz/PycharmProjects/vleo-bench/data/RSICD/qwen_rsicd_captions.jsonl", lines=True)

        def parse_response(x):
            if any([keyword in x for keyword in ["The image", "appears", "is", "Therefore", "Can you", "The given image"]]):
                return "-1"
            try:
                return x.replace(": ", ". ").split(". ", maxsplit=2)[-1].rstrip(".")
            except IndexError:
                return "-1"

        # print(df["response"].str.split(". "))
        df["classification"] = df["response"].apply(parse_response)
        print(df["response"].apply(lambda x: x.replace(": ", ". ").split(". ", maxsplit=2)))
        # del df["response"]

        df["label"] = df["img"].replace(self.img2landcover)
        df["caption"] = df_caption["response"]
        df["ref_captions"] = [json.dumps(x) for x in self.convert_hf_dataset("val")["captions"]]

        print(classification_report(y_pred=df["classification"], y_true=df["label"]))

        df.to_csv("/home/danielz/PycharmProjects/vleo-bench/data/RSICD/qwen_rsicd.csv", index=False)

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
    print(dataset.get_multiple_choice_prompts())
    print(dataset.land_cover_types)
    hf_dataset = dataset.convert_hf_dataset("val")

    result_path = "/home/danielz/PycharmProjects/vleo-bench/data/RSICD/kosmos-2.jsonl"
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
        if type_count[land_cover_type] > 6:
            continue

        fpath = os.path.join(dataset.data_dir, dataset.image_dir, data_item["path"])
        assert os.path.exists(fpath)
        query_payload, query_result = dataset.query_hf(fpath)
        print(data_item["path"], query_result)  # query_result["choices"][0]["message"]["content"])

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
    print(dataset.get_multiple_choice_prompts())
    dataset.parse_qwen()


if __name__ == "__main__":
    validate()
