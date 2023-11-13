import os
from abc import ABC

import openai
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

    def download_url(self, url: str):
        pass

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def query_openai(self, base64_image: str, system: str, user: str):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "OpenAI-Organization": os.environ["OPENAI_API_ORG"]
        }

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
