import os
import importlib
import re
from typing import Dict, Any, List
from datetime import datetime
from HiveMinds.minds.mind import Mind
import inspect
import json

import signal
from typing import Dict, Any, List, Optional

import ast

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

# tool_result = execute_tool(config['Tool_Dict'], tool_name, tool_params_kwargs)

class Executor(Mind):
    ROLE = "Executor"
    # def __init__(self, 
    #              toolkits: Dict[str, Any],
    #              root_cache_dir: str = "solver_cache",  
    #              verbose: bool = False):
    def __init__(self, 
                 role: str, instruction: str, 
                 toolkits: Dict[str, Any], model_info='deepseek-chat', 
                 root_cache_dir: str = "solver_cache",  
                 is_recorded=False, is_continue=False, 
                 verbose: bool = False):
        super().__init__(
            role=role if role else self.ROLE,  
            instruction=instruction, 
            model_info=model_info,
            is_recorded=is_recorded,
            is_continue=is_continue
        )
    
    
        self.toolkits = toolkits
        self.root_cache_dir = root_cache_dir
        # self.request_llm = 
        
        self.verbose = verbose

    def set_query_cache_dir(self, query_cache_dir):
        if query_cache_dir:
            self.query_cache_dir = query_cache_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.query_cache_dir = os.path.join(self.root_cache_dir, timestamp)
        os.makedirs(self.query_cache_dir, exist_ok=True)
        
    def request_llm(self, system_prompt: str, user_prompt: str, last_only=True, **kwargs) -> str:
        think_content, answer_content = self.chat(user_prompt, instruction=system_prompt)
        
        self._save_token_file(os.path.join(os.environ["OUTPUT_DIR"], "tool_token_count.json"), self.token_count)
        
        program = ""
        start = False
        for line in answer_content.strip().split("\n"):
            if line.startswith("```python") or line.startswith("python"):
                if last_only:
                    program = ""  # 只提取最后一个程序
                else:
                    program += "\n# ========\n"
                start = True
            elif line.startswith("```"):
                start = False
            elif start:
                program += line + "\n"
        return program
        
    
    def execute_tool(self, tool_name: str, kwargs: Dict[str, Any]):
        """
        执行工具，返回一个字典，包含工具的返回结果。
        :param tool_name: 待调用的工具名称
        :param kwargs: 待调用工具的参数字典
        :return:
        """
        tool_func = self.toolkits[tool_name]
        # 如果没有找到工具，输出错误
        if tool_func is None:
            raise ValueError(f"Tool {tool_name} Not Found in Toolset")
        
        ################################################
        # TODO: 
        sig = inspect.signature(tool_func)
        # 获取工具函数的所有参数名（仅关键字参数或位置+关键字参数）
        all_param_names = [
            param.name for param in sig.parameters.values()
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        ]
        # 核心逻辑：判断工具函数是否需要 request_llm，若需要则加入 kwargs
        if 'request_llm' in all_param_names:
            kwargs['request_llm'] = self.request_llm
        ###################################################################

        return_value = tool_func(**kwargs)

        # 根据返回值是否为元组，生成对应的结果字典
        if isinstance(return_value, tuple):
            result_dict = {}
            for i, val in enumerate(return_value):
                if isinstance(val, list):
                    result_dict[f"return{i + 1}"] = {
                        "type": [type(item).__name__ for item in val],
                        "value": val
                    }
                else:
                    result_dict[f"return{i + 1}"] = {"type": type(val).__name__, "value": val}
        elif isinstance(return_value, list):
            result_dict = {"return1": {"type": [type(item).__name__ for item in return_value], "value": return_value}}
        else:
            result_dict = {"return1": {"type": type(return_value).__name__, "value": return_value}}

        return result_dict
    
    # TODO: 保存token_count到文件
    def _save_token_file(self, checkpoint_path: str, new_data: Dict, save_interval: int = 10):
        
        try:
            # 读取现有数据（若文件不存在则初始化空列表）
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint: Dict = json.load(f)
            except FileNotFoundError:
                checkpoint = {}
            
            checkpoint[len(checkpoint)] = new_data
            
            # checkpoint = update_json_file(checkpoint, new_data)

            # 原子性写入：先写入临时文件，再替换原文件
            with open(f"{checkpoint_path}.tmp", "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=4, ensure_ascii=False)
            
            # 替换原文件（确保写入完整）
            # import os
            os.replace(f"{checkpoint_path}.tmp", checkpoint_path)
            # print(f"Checkpoint saved at {checkpoint_path} (Total entries: {len(checkpoint_list)})")
        
        except Exception as e:
            print(f"保存检查点失败: {e}")
            # 可添加重试逻辑或忽略错误继续训练
        

