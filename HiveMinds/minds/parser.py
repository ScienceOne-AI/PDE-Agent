import os
import re
from PIL import Image
from typing import Dict, Any, List, Tuple
import json
import logging

# from HiveMinds.memory import Memory
from HiveMinds.minds.mind import Mind
from HiveMinds.context.build_context import build_tool_use_prompt, build_tool_use_prompt_false

# from octotools.engine.factory import create_llm_engine
# from octotools.models.memory import Memory
# from octotools.models.formatters import QueryAnalysis, NextStep, MemoryVerification

class Parser(Mind):
    ROLE = "Parser"
    def __init__(self, role: str, instruction: str, 
                 model_info='deepseek-chat', 
                 is_recorded=False, is_continue=False, 
                 img_path=None, verbose: bool = False):
        super().__init__(
            role=role if role else self.ROLE, 
            instruction=instruction, 
            model_info=model_info,
            is_recorded=is_recorded,
            is_continue=is_continue
        )
        # self.toolkit_info = toolkit_info
        # self.toolbox_metadata = toolbox_metadata if toolbox_metadata is not None else {}
        # self.available_tools = available_tools if available_tools is not None else []
        self.verbose = verbose
        
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        image_info = {}
        if image_path and os.path.isfile(image_path):
            image_info["image_path"] = image_path
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                image_info.update({
                    "width": width,
                    "height": height
                })
            except Exception as e:
                print(f"Error processing image file: {str(e)}")
        return image_info

    def generate_base_response(self, question: str, image: str, max_tokens: str = 4000) -> str:
        image_info = self.get_image_info(image)

        input_data = [question]
        if image_info and "image_path" in image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        self.base_response = self.llm_engine_mm(input_data, max_tokens=max_tokens)

        return self.base_response
    
    def unpack_tool_params(self, tool_params, previous_actions):
        """
        """
        kwargs = {}
        conn_action_step = set()
        step_pattern = re.compile(r"Step_(\d+)_(\d+)")  # 匹配 "Step_n_m" 格式

        for param_name, param_info in tool_params.items():
            param_value = param_info["value"]

            # 仅当参数值为字符串时检查是否为 "Step_n_m" 格式
            if isinstance(param_value, str):
                match = step_pattern.fullmatch(param_value)
                if match:
                    conn_action_step.add(int(match.group(1)))  # 记录连接步骤
                    step_index = int(match.group(1)) - 1  # 转换为列表索引（从 0 开始）
                    return_key = f"return{int(match.group(2))}"  # 构造返回值键，例如 "return1"
                    try:
                        param_value = previous_actions[step_index]['return'][return_key]["value"]
                    except (IndexError, KeyError):
                        raise ValueError(f"Invalid reference: {param_value} not found in previous_steps_info")

            kwargs[param_name] = param_value

        return kwargs, conn_action_step


    def process_tool_use(self, 
                        # question: str, 
                        tool_info,      # tuple (tool_name, tool_info)
                        previous_actions,
                        false_action=None,
                        action_item=None) -> str:
        if false_action is None:
            tool_use_prompt = build_tool_use_prompt(tool_info, previous_actions).strip()
        else:
            tool_use_prompt = build_tool_use_prompt_false(tool_info, previous_actions, false_action,  action_item).strip()
        
        tool_use_think, tool_use_answer = self.chat(tool_use_prompt)
        
        # 工具解析
        tool_params = json.loads(tool_use_answer)

        self.memory.set_current_info({
            'tool_name': tool_info[0],
            'params': tool_params
        })
        
        # logging
        logging.info(f"Tool use prompt: \n{tool_use_prompt}")
        logging.info(f"Tool use think: \n{tool_use_think}")
        logging.info(f"Tool use answer: \n{tool_use_answer}")
        # logging.info(f"Tool params kwargs: \n{tool_params_kwargs}")
        # logging.info(f"Connection action step: \n{conn_action_step}")
        logging.info(f"Tool Params: \n{tool_params}")
        
        # return tool_params_kwargs, conn_action_step
        return tool_params

    
    #  get tokens
    def get_current_tokens(self):
        return self.current_token
    