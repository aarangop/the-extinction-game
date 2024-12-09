from llama_cpp import Llama
import pandas as pd
from typing import Dict, Union
import json
import re


def extract_json_from_llm_response(response: str):
    """
    Extracts the JSON from the response of the LLM API
    """
    # Extract the JSON from the response
    json_response = re.search(r'\{.*\}', response).group()
    json_obj = json.loads(json_response)
    return json_obj
