try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install the openai package by running `pip install openai`, and add 'DEEPSEEK_API_KEY' to your environment variables.")

import os
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    retry_if_result,
)
from openai import APIError

from typing import List, Union
from .base import EngineLM, CachedEngine


class S1(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

    def __init__(
        self,
        model_info="s1",
        use_cache: bool=False,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool=False):

        self.model_info = model_info
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal
        # TODO: add tokens
        self.current_token = {}

        # self.is_chat_model = any(x in model_info for x in ["deepseek-chat"])
        # self.is_reasoning_model = any(x in model_info for x in ["deepseek-reasoner"])
        self.is_chat_model = True
        self.is_reasoning_model = False

        if self.use_cache:
            root = platformdirs.user_cache_dir("HiveMinds")
            cache_path = os.path.join(root, f"cache_deepseek_{model_info}.db")
            super().__init__(cache_path=cache_path)

        if os.getenv("S1_API_KEY") is None:
            raise ValueError("Please set the S1_API_KEY environment variable.")
        self.client = OpenAI(
            api_key=os.getenv("S1_API_KEY"),
            base_url="https://uni-api.cstcloud.cn/v1"
        )
        
    def _get_default_message(self):
        return [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": ""},
                ]

    # 等待策略：指数退避 + 随机抖动
    # @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    @retry(
        wait=wait_random_exponential(min=1, max=5),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, APIError))
    )
    def generate(
        self, message=None, temperature=0, max_tokens=5120, top_p=0.8, response_format=None
    ):
        # sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        message = message or self._get_default_message()

        if self.use_cache:
            # cache_key = sys_prompt_arg + prompt
            cache_key = message
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none
        
        if self.is_chat_model:
            response = self.client.chat.completions.create(
                model=self.model_info,
                messages=message,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                extra_body={
                    "repetition_penalty": 1.05,
                },
            )
            
            # add tokens
            usage = dict(response.usage) if hasattr(response, 'usage') else {}
            # 安全地提取各类型token数量（确保为整数类型）
            input_tokens = int(usage.get("prompt_tokens", 0))
            output_tokens = int(usage.get("completion_tokens", 0))
            total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens))
            # 更新当前token使用状态
            self.current_token = {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens
            }
            
            response = response.choices[0].message.content

        elif self.is_reasoning_model:
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=message,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body={
                    "repetition_penalty": 1.05,
                },
            )
            response = response.choices[0].message.content

        if self.use_cache:
            self._save_cache(cache_key, response)
        return response

    def __call__(self, message=None, **kwargs):
        return self.generate(message=message, **kwargs)
    
    def get_current_tokens(self):
        return self.current_token
    