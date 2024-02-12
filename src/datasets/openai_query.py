import json
import os
from abc import ABC
from typing import List, Union
import datetime
import asyncio
import openai
import pandas as pd
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from src.utils.gpt4v_chat import base64_image_to_bytes
from src.datasets.VisualChat_abc import VisualChat
openai.api_key = os.getenv("OPENAI_API_KEY")

LLAMA2_TEMPLATE = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_msg_1} [/INST]
"""


class OpenAIDataset(VisualChat):
    model = None
    tokenizer = None
    pipeline = None
    system_message = ("You are a helpful image analyst that specializes in satellite and aerial images. You always "
                      "truthfully answer the user's question. If you are not sure about something, don't answer false"
                      " information.")
    openai_api_org = None
    openai_api_key = None
    start_time = int(datetime.datetime.utcnow().timestamp())

    def __init__(self, credential_path: str):
        super(VisualChat, self).__init__()

        credentials = []

        assert credential_path.endswith(".jsonl")
        with open(credential_path) as f:
            for line in f.readlines():
                credentials.append(json.loads(line.strip()))
            self.credentials = pd.DataFrame(credentials).set_index("OpenAI-Organization")
            if "usage" not in self.credentials.columns:
                self.credentials["usage"] = 0

        self.credential_path = credential_path
        self.get_credentials()

    def _update_usage(self):
        # Save the credentials to a new file based on the start time
        self.credentials.reset_index().to_json(
            self.credential_path.replace(".jsonl", f"_{self.start_time}.jsonl"),
            lines=True,
            orient="records"
        )

    def download_url(self, url: str):
        pass

    def get_credentials(self):
        if (self.openai_api_org and
                self.credentials.groupby(["OpenAI-Organization"]).first().loc[
                    self.openai_api_org, "usage"] < 500):
            return self.openai_api_key, self.openai_api_org

        if os.environ.get("OPENAI_API_ORG"):
            openai_api_org = os.environ["OPENAI_API_ORG"]
            openai_api_key = os.environ["OPENAI_API_KEY"]
        else:
            available_orgs = self.credentials[self.credentials["usage"] < 500].sort_values(by="usage", ascending=True)
            openai_api_org = available_orgs.index.tolist()[0]
            openai_api_key = available_orgs.iloc[0]["key"]

            self.openai_api_org = openai_api_org
            self.openai_api_key = openai_api_key
            print(f"Switching to new org {openai_api_org}")
            self._update_usage()

        return openai_api_key, openai_api_org
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def query(self, base64_image: Union[List[str], str], system: str, user: str, t: float = 0.2,
                     max_tokens: int = 300,
                     demos=None):
        if demos is None:
            demos = []

        if isinstance(base64_image, str):
            base64_image = [base64_image]

        openai_api_key, openai_api_org = self.get_credentials()

        # used for HTTP request to the OpenAI API
        # Content-Type
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
            "OpenAI-Organization": openai_api_org
        }

        print(system + "\n" + user)

        if len(base64_image) == 1:
            user_message = [
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
        else:
            user_message = [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{x}",
                    "detail": "high"
                }
            } for x in base64_image] + [
                               {
                                   "type": "text",
                                   "text": user
                               },
                           ]

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
                    "content": user_message
                }
            ],
            "max_tokens": max_tokens,
            "temperature": t
        }
        # makes request to OpenAI API 
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        remaining_requests = int(response.headers["x-ratelimit-remaining-requests"])

        reset_time = response.headers["x-ratelimit-reset-requests"]
        if not response.ok:
            if "RPD" in str(response.json()):
                print(f"Rate limit exceeded. Reset in {reset_time}\n" + str(response.json()))
                self.credentials.loc[openai_api_org, "usage"] = 500
                retry_response = self.query(base64_image, system, user, t, max_tokens, demos)
                return retry_response
            else:
                raise IOError(str(response.json()))
        else:
            reset_time = response.headers["x-ratelimit-reset-requests"]
            print(f"{remaining_requests} queries left for {openai_api_org}. Reset in {reset_time}")
            self.credentials.loc[openai_api_org, "usage"] = 500 - remaining_requests
            self._update_usage()

        return payload, response.json()