import os
import logging
import json
import sys
from typing import Dict
import re

def setup_logging(log_file):
    """配置日志系统，支持多次重新配置"""
    root_logger = logging.getLogger()
    
    # 清除所有现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 定义统一的日志格式
    formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s\n', datefmt='%H:%M:%S')
    
    # 配置文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 配置控制台处理器
    # 配置控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 设置日志级别
    root_logger.setLevel(logging.INFO)
    
def init_file(user_input, file_path='', difficulty=None):
    """initialize the execution message file"""
    with open(os.path.join(os.environ["OUTPUT_DIR"], "execution_message_step.json"), "w", encoding="utf-8") as f:
        json.dump({
            "input_file": file_path,
            "problem_description": user_input,
            "difficulty_level": difficulty,
            "RAG_case": None,
            "planning_task": None,
            "execute_tool_chain": [],
            "valid_steps_info": [],
            "total_token": 0,
            }, f, ensure_ascii=False, indent=4)
    
    
def update_json_file(checkpoint: Dict, new_data: Dict):
    for key, value in new_data.items():
        # TODO: 处理嵌套字典，处理不同格式数据
        # if key in checkpoint:
        #     if isinstance(checkpoint[key], dict) and isinstance(value, dict):
        #         checkpoint[key] = update_json_file(checkpoint[key], value)
        #     elif isinstance(checkpoint[key], list) and isinstance(value, list):
        #         checkpoint[key].extend(value)
        #     else:
        #         checkpoint[key] = value
        # else:
        #     checkpoint[key] = value
        
        # 直接更新
        if key in checkpoint and isinstance(checkpoint[key], list):
            if key == "execute_tool_chain":
                # replace the tool info
                
                # if checkpoint[key]:
                #     for i, tool_info in enumerate(checkpoint[key]):
                #         if tool_info['tool_name'] == value['tool_name']:
                #             checkpoint[key][i] = value
                # else:
                #     checkpoint[key].append(value)
                try:
                    # 快速定位第一个匹配项的索引
                    index = next(i for i, d in enumerate(checkpoint[key]) if "tool_name" in d and d["tool_name"] == value.get('tool_name', ''))
                    checkpoint[key][index] = value  # 替换
                except StopIteration:
                    checkpoint[key].append(value)  # 未匹配到，添加
                
            else:
                checkpoint[key].append(value)
        else:
            checkpoint[key] = value
            
    return checkpoint

def replace_or_add(lst, key, target_value, new_element):
    """
    匹配到元素则替换，未匹配到则添加新元素
    
    参数:
        lst: 包含字典的列表
        key: 用于匹配的键
        target_value: 目标值
        new_element: 用于替换或添加的元素
    
    功能:
        1. 查找列表中第一个满足 `d[key] == target_value` 的字典
        2. 找到则替换为新元素；未找到则添加新元素到列表末尾
        3. 原地修改列表，无返回值（或可返回修改后的列表）
    """
    try:
        # 快速定位第一个匹配项的索引
        index = next(i for i, d in enumerate(lst) if key in d and d[key] == target_value)
        lst[index] = new_element  # 替换
    except StopIteration:
        lst.append(new_element)  # 未匹配到，添加
    
def save_checkpoint(new_data: Dict, checkpoint_path: str='', save_interval: int = 10):
    """
    保存检查点到JSON文件，支持增量更新和原子性写入。
    
    参数：
    - checkpoint_path: 检查点文件路径（如 "training_logs.json"）
    - new_data: 本轮训练生成的新字典（需已处理非JSON类型）
    - save_interval: 保存间隔（每n轮保存一次，避免频繁写入）
    """
    try:
        if not checkpoint_path:
            checkpoint_path = os.path.join(os.getenv('OUTPUT_DIR'), "execution_message_step.json")
        
        # 读取现有数据（若文件不存在则初始化空列表）
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint: Dict = json.load(f)
        except FileNotFoundError:
            checkpoint = {}
        
        # 预处理新数据（确保可序列化，可选：若训练中已处理可跳过）
        # new_data = replace_unsupported(new_data)  # 若已处理，注释此行
        
        # # 添加新数据到列表
        # checkpoint_list.append(new_data)
        # 根据传入输入修改json文件，自定义一个函数
        checkpoint = update_json_file(checkpoint, new_data)
        
        
        # 原子性写入：先写入临时文件，再替换原文件
        with open(f"{checkpoint_path}.tmp", "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=4, ensure_ascii=False)
        
        # 替换原文件（确保写入完整）
        
        os.replace(f"{checkpoint_path}.tmp", checkpoint_path)
        # print(f"Checkpoint saved at {checkpoint_path} (Total entries: {len(checkpoint_list)})")
    
    except Exception as e:
        print(f"保存检查点失败: {e}")
        # 可添加重试逻辑或忽略错误继续训练
        
def is_json_supported(value):
    """判断值是否为 JSON 支持的类型"""
    supported_types = (str, int, float, bool, list, dict, type(None))
    return isinstance(value, supported_types) or isinstance(value, tuple)  # tuple 可转为 JSON 数组


def replace_unsupported(obj):
    """递归替换非 JSON 支持的类型为格式化字符串"""
    # 处理字典（键值对结构）
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            # 递归处理值，并判断是否替换
            new_value = replace_unsupported(value)
            new_dict[key] = new_value
        return new_dict
    
    # 处理列表/元组（有序元素集合）
    elif isinstance(obj, (list, tuple)):
        return [replace_unsupported(item) for item in obj]
    
    # 处理其他类型：非 JSON 支持的类型进行替换
    else:
        # return f"This is a {type(obj).__name__} object" if not is_json_supported(obj) else obj
        return f"This is a {type(obj).__module__ + '.' + type(obj).__name__} object" if not is_json_supported(obj) else obj
        # return f"This is a {obj.__class__.__module__ + '.' + obj.__class__.__qualname__} object" if not is_json_supported(obj) else obj
        

def update_step_info(current_step_info: Dict, tool_params: Dict):
    """

    Args:
        current_step_info (Dict): 
            :param current_step_info: 一个字典，包含工具名称、参数信息和返回结果，如：
            current_step_info = {
                "step": idx+1,
                "tool_name": tool_name,
                "params": {"param_name": value, ...},
                "return": {**tool_result}
            }
        tool_params (Dict): 
            :param tool_params: 工具所需参数的字典，格式如：
            {
                "param_name": {
                    "type": "类型描述",
                    "value": "Step_n_m" 或直接值
                },
                ...
            }
        Returns:
            Dict: _description_
    """
    
    step_pattern = re.compile(r"Step_(\d+)_(\d+)")  # 匹配 "Step_n_m" 格式
    current_step_info = replace_unsupported(current_step_info)
    
    for param_name, param_info in tool_params.items():
        param_value = param_info["value"]

        # 仅当参数值为字符串时检查是否为 "Step_n_m" 格式
        if isinstance(param_value, str):
            match = step_pattern.fullmatch(param_value)
            if match and current_step_info["params"].get(param_name, None):
                if isinstance(current_step_info["params"][param_name], list):
                    # current_step_info["params"][param_name].append(f" at {match.group()}")
                    current_step_info["params"][param_name] = [item + f" at {match.group()}" for item in current_step_info["params"][param_name]]
                else:
                    current_step_info["params"][param_name] += f" at {match.group()}"

    return current_step_info

def adjust_execution_file(loop_count, turn_count=None, checkpoint_path=''):
    try:
        if not checkpoint_path:
            checkpoint_path = os.path.join(os.getenv('OUTPUT_DIR'), "execution_message_step.json")
        
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if turn_count:
            data[f'execute_tool_chain_{loop_count}_{turn_count}'] = data['execute_tool_chain']
            data[f'valid_steps_info_{loop_count}_{turn_count}'] = data['valid_steps_info']
            data['valid_steps_info'] = []
        else:
            data[f'execute_tool_chain_{loop_count}'] = data['execute_tool_chain']
            data[f'valid_steps_info_{loop_count}'] = data['valid_steps_info']
            data[f"planning_task_{loop_count}"] = data['planning_task']
            data['execute_tool_chain'] = []
            data['valid_steps_info'] = []
            data['planning_task'] = None
        
        # 文件原子写入
        # with open(os.path.join(os.environ["OUTPUT_DIR"], "execution_message_step.json"), "w", encoding="utf-8") as f:
        #     json.dump(data, f, ensure_ascii=False, indent=4)
            
        # 原子性写入：先写入临时文件，再替换原文件
        with open(f"{checkpoint_path}.tmp", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        # 替换原文件（确保写入完整）
        os.replace(f"{checkpoint_path}.tmp", checkpoint_path)
        return True
    except Exception as e:
        print(f"保存检查点失败: {e}")
        # 可添加重试逻辑或忽略错误继续训练
        return False
