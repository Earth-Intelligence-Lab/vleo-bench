import json
import os
from abc import ABC, abstractmethod
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

openai.api_key = os.getenv("OPENAI_API_KEY")

LLAMA2_TEMPLATE = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_msg_1} [/INST]
"""


class VisualChat(ABC):
    @abstractmethod
    def __init__(self, credential_path: str):
        '''
        Given a path to a jsonl file, load the credentials into a pandas dataframe

        Args:
            credential_path (str): The path to the jsonl file. Must end with .jsonl

        Returns:
            None
        '''
        pass

    @abstractmethod
    def get_credentials(self):
        '''
        Get the credentials from the file
        '''
        pass

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    @abstractmethod
    def query(self, base64_image: Union[List[str], str], system: str, user: str, t: float = 0.2,
                     max_tokens: int = 300,
                     demos=None):
        pass