import os
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from transformers import AutoTokenizer
import transformers
import torch

openai.api_key = os.getenv("OPENAI_API_KEY")


LLAMA2_TEMPLATE = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_msg_1} [/INST]
"""


class VLEODataset:
    model = "meta-llama/Llama-2-70b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    def download_url(self, url: str):
        pass

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def request_openai(self, system: str, user: str):
        return openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            top_p=0.1,
            temperature=0.2,
            max_tokens=256
        )

    def request_loca_llama(self, system: str, user: str):
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
