from huggingface_hub import InferenceClient
from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from langchain_core.language_models import BaseChatModel
import json
import datasets
import os

class BaseLLM():
    def __init__(self, **config):
        self.config = config

# llamacpp, ollama, vllm, lmdeploy
class OpenAIAPILikeLLM(BaseLLM):
    def __init__(self, **config):
        super().__init__(**config)
        from openai import OpenAI

        # if "api_key" in config:
        #     self.api_key = config["sk-4c3418cc8bac4b39bea040bc4e6a4d3b"]
        # else:
        self.api_key = os.environ.get("LLM_API_KEY", "Bearer no-key")

        self.model = self.config.get("model", "llama3")
        self.stop = self.config.get("stop", None)
        self.stream = self.config.get("stream", True)
        self.base_url = self.config.get("base_url", "http://192.168.101.15:8081/v1")
        self.function = self.config.get("function", False)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.temperature = self.config.get("temperature", 0)
        self.max_tokens = self.config.get("max_tokens", 512)

    def generate(
        self,
        chat_history,
        **kwargs,
    ):
        response = self.client.chat.completions.create(
            messages=chat_history,
            model=kwargs.get("model", self.model),
            stream=self.stream,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            stop=kwargs.get("stop", self.stop),
        )

        yield from response

