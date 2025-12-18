import os
import re
from typing import List, Dict, Optional
import ast




class Resources_Pool:
    def __init__(self, resources_path: str=None):
        self.resources_path = resources_path
        self.resources = {}
        
    def add_resource(self, step_idx: int, resource: Dict) -> None:
        if isinstance(resource, dict):
            for idx, (key, value) in enumerate(resource.items()):
                self.resources[f"Step_{step_idx}_{idx+1}"] = value['value']
           
    def refress_resources(self, resource: Dict) -> None:
        if isinstance(resource, dict):
            self.resources.update(resource)
           
    def get_resource(self, resource_name: str):
        if resource_name in self.resources:
            return self.resources[resource_name]
        else:
            return None
    
    def get_all_resources(self) -> Dict:
        return self.resources
    
    def eliminate_false_resources(self, step_idx: list):
        startwith_tuple = tuple(f"Step_{idx}_" for idx in step_idx)
        for key in list(self.resources.keys()):
            # if any(key.startswith(startwith) for startwith in startwith_tuple):
            if key.startswith(startwith_tuple):
                self.resources.pop(key)
                
    def clear(self):
        self.resources = {}
            
    
    def __call__(self, resource_name, **kwds):
        return self.get_resource(resource_name)
    
    # FIXME: use to gpt-4o-mini
    def first_step_numbers(self, obj):
        """
        返回 (step_id, first_number) 或 (None, None)
        step_id  : 如 "Step_3_1"
        first_number : 如 3
        """
        # 统一成字符串列表
        if isinstance(obj, list):
            seq = [str(item) for item in obj]
        elif isinstance(obj, str):
            try:
                parsed = ast.literal_eval(obj)
                seq = [str(item) for item in parsed] if isinstance(parsed, list) else [obj]
            except (ValueError, SyntaxError):
                seq = [obj]
        else:
            return None, None

        pattern = re.compile(r'(Step_(\d+)_\d+)')
        for text in seq:
            m = pattern.search(text)
            if m:
                return m.group(1), int(m.group(2))
        return None, None
    
    def unpack_tool_params(self, tool_params, tool_name, previous_actions=None):
        """
        Parser tool parameters and return a kwargs dict for calling the tool.
        """
        kwargs = {}
        conn_action_step = set()
        step_pattern = re.compile(r"Step_(\d+)_(\d+)")  # 匹配 "Step_n_m" 格式

        for param_name, param_info in tool_params.items():
            param_value = param_info["value"]

            # 仅当参数值为字符串时检查是否为 "Step_n_m" 格式
            # if isinstance(param_value, str):
            #     match = step_pattern.fullmatch(param_value)
            #     if match:
            #         conn_action_step.add(int(match.group(1)))  # 记录连接步骤
            #         # step_index = int(match.group(1)) - 1  # 转换为列表索引（从 0 开始）
            #         # return_key = f"return{int(match.group(2))}"  # 构造返回值键，例如 "return1"
            #         match_key = match.group(0)
            #         try:
            #             # param_value = previous_actions[step_index]['return'][return_key]["value"]
            #             param_value = self.get_resource(match_key)
            #         except (IndexError, KeyError):
            #             raise ValueError(f"Invalid reference: {param_value} not found in previous_steps_info")
            ## TODO: 转化为str看看有没有
            if isinstance(param_value, str):
                if param_value.strip().lower() in {'none', 'null', ''}:
                    param_value = None
            
            search_rst, search_step = self.first_step_numbers(param_value)
            if search_rst:
                try:
                    # param_value = previous_actions[step_index]['return'][return_key]["value"]
                    param_value = self.get_resource(search_rst)
                    conn_action_step.add(search_step)
                except (IndexError, KeyError):
                    raise ValueError(f"Invalid reference: {param_value} not found in previous_steps_info")

            kwargs[param_name] = param_value

        return kwargs, conn_action_step