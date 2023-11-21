import json
import os
from abc import ABC
from typing import List
import datetime

import openai
import pandas as pd
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

openai.api_key = os.getenv("OPENAI_API_KEY")

LLAMA2_TEMPLATE = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_msg_1} [/INST]
"""


class VLEODataset(ABC):
    model = None
    tokenizer = None
    pipeline = None
    system_message = ("You are a helpful image analyst that specializes in satellite and aerial images. You always "
                      "truthfully answer the user's question. If you are not sure about something, don't answer false"
                      " information.")
    openai_api_org = None
    openai_api_key = None

    def __init__(self, credential_path: str):
        credentials = []

        assert credential_path.endswith(".jsonl")
        with open(credential_path) as f:
            for line in f.readlines():
                credentials.append(json.loads(line.strip()))
            self.credentials = pd.DataFrame(credentials).set_index("OpenAI-Organization")
            if "usage" not in self.credentials.columns:
                self.credentials["usage"] = 0

        self.credential_path = credential_path
        self.get_openai_credentials()

    def _update_usage(self):
        self.credentials.reset_index().to_json(
            self.credential_path.replace(".jsonl", f"_{int(datetime.datetime.utcnow().timestamp())}.jsonl"),
            lines=True,
            orient="records"
        )

    def download_url(self, url: str):
        pass

    def get_openai_credentials(self):
        if (self.openai_api_org and
                self.credentials.groupby(["OpenAI-Organization"]).first().loc[
                    self.openai_api_org, "usage"] < 100):
            return self.openai_api_key, self.openai_api_org

        if os.environ.get("OPENAI_API_ORG"):
            openai_api_org = os.environ["OPENAI_API_ORG"]
            openai_api_key = os.environ["OPENAI_API_KEY"]
        else:
            available_orgs = self.credentials[self.credentials["usage"] < 100].sort_values(by="usage", ascending=True)
            openai_api_org = available_orgs.index.tolist()[0]
            openai_api_key = available_orgs.iloc[0]["key"]

            self.openai_api_org = openai_api_org
            self.openai_api_key = openai_api_key
            print(f"Switching to new org {openai_api_org}")
            self._update_usage()

        return openai_api_key, openai_api_org

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def query_openai(self, base64_image: str, system: str, user: str, t: float = 0.2, max_tokens: int = 300,
                     demos=None):
        if demos is None:
            demos = []

        openai_api_key, openai_api_org = self.get_openai_credentials()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
            "OpenAI-Organization": openai_api_org
        }

        # print(system + "\n" + user)

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system
                        }
                    ]
                },
                *demos,
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": t
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if not response.ok:
            if "RPD" in str(response.json()):
                print("Rate limit exceeded\n" + str(response.json()))
                self.credentials.loc[openai_api_org, "usage"] = 100
                retry_response = self.query_openai(base64_image, system, user, t, max_tokens, demos)
                return retry_response
            else:
                raise IOError(str(response.json()))
        else:
            self.credentials.loc[openai_api_org, "usage"] += 1

        return payload, response.json()

    def request_loca_llama(self, model: str, system: str, user: str):
        from transformers import AutoTokenizer
        import transformers
        import torch

        if self.pipeline is None or model != self.model:
            self.model = model
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        sequences = self.pipeline(
            LLAMA2_TEMPLATE.format(system_prompt=system, user_msg_1=user),
            do_sample=True,
            top_p=0.4,
            top_k=50,
            temperature=0.2,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=256,
        )

        return sequences[0]["generated_text"]
