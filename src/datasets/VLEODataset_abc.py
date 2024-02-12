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

'''
Constructs a VLEO dataset object that can later be used to query the OpenAI API for responses to user questions
'''
class VLEODataset(ABC):
    
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
