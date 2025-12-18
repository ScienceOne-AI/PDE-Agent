from typing import Dict, Any, List, Union, Optional
import os

class Memory:

    def __init__(self):
        self.query: Optional[str] = None
        self.files: List[Dict[str, str]] = []
        self.actions: Dict[str, Dict[str, Any]] = {}
        self._init_file_types()
        self.memory_info = {}   # Dict[str, Dict[str, Any]]
        self.messages = []
        self.current_info = None
        
    def add_memory_info(self, info: str) -> None:
        self.memory_info.update(info)
        self.current_info = info
        # if not self.current_info:
        #     self.current_info = info

    def add_message(self, message: str) -> None:
        self.messages.append(message)
        
    def clear_memory(self) -> None:
        self.memory_info = {}
        self.messages = []
        
    def refresh_memory(self, info) -> None:
        self.current_info = info
        
    def set_current_info(self, info: str) -> None:
        self.current_info = info
            
    def eliminate_false_memory(self, eliminate_mem_keys: List[str]):
        self.false_memory = {}
        for key in eliminate_mem_keys:
            self.false_memory.update(self._clear_from_key(key))
        return self.false_memory
        
    def share_memory_all(self) -> None:
        pass
    
    def _clear_from_key(self, key):
        """清空字典中指定键的值，根据类型执行不同的清空策略"""
        if key not in self.memory_info:
            return {}  # 键不存在，直接返回原字典
        
        value = self.memory_info[key].copy()  # 保存原值
        
        # 根据值的类型执行不同的清空操作
        if isinstance(value, list):
            self.memory_info[key] = []  # 清空列表
        elif isinstance(value, dict):
            self.memory_info[key] = {}  # 清空字典
        elif isinstance(value, set):
            self.memory_info[key] = set()  # 清空集合
        elif isinstance(value, str):
            self.memory_info[key] = ""  # 清空字符串
        elif isinstance(value, (int, float, bool)):
            # 数字和布尔值无法"清空"，通常重置为0或False
            self.memory_info[key] = 0 if isinstance(value, (int, float)) else False
        else:
            # 其他类型（如自定义对象）：设为None或调用对象的清空方法
            try:
                # 尝试调用对象自身的清空方法（如果有）
                self.memory_info.clear()
            except AttributeError:
                self.memory_info[key] = None  # 默认设为None
        
        # 返回清空前的值
        return {key: value}
    
    def get_memory_info(self) -> Dict[str, Any]:
        return self.memory_info
    
    def get_memory_values(self) -> List[str]:
        return list(self.memory_info.values())
    
    def get_messages(self) -> List[str]:
        return self.messages
    
    def get_current_info(self) -> str:
        return self.current_info

    def set_query(self, query: str) -> None:
        if not isinstance(query, str):
            raise TypeError("Query must be a string")
        self.query = query

    def _init_file_types(self):
        self.file_types = {
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
            'text': ['.txt', '.md'],
            'document': ['.pdf', '.doc', '.docx'],
            'code': ['.py', '.js', '.java', '.cpp', '.h'],
            'data': ['.json', '.csv', '.xml'],
            'spreadsheet': ['.xlsx', '.xls'],
            'presentation': ['.ppt', '.pptx'],
        }
        self.file_type_descriptions = {
            'image': "An image file ({ext} format) provided as context for the query",
            'text': "A text file ({ext} format) containing additional information related to the query",
            'document': "A document ({ext} format) with content relevant to the query",
            'code': "A source code file ({ext} format) potentially related to the query",
            'data': "A data file ({ext} format) containing structured data pertinent to the query",
            'spreadsheet': "A spreadsheet file ({ext} format) with tabular data relevant to the query",
            'presentation': "A presentation file ({ext} format) with slides related to the query",
        }

    def _get_default_description(self, file_name: str) -> str:
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()

        for file_type, extensions in self.file_types.items():
            if ext in extensions:
                return self.file_type_descriptions[file_type].format(ext=ext[1:])

        return f"A file with {ext[1:]} extension, provided as context for the query"
    
    def add_file(self, file_name: Union[str, List[str]], description: Union[str, List[str], None] = None) -> None:
        if isinstance(file_name, str):
            file_name = [file_name]
        
        if description is None:
            description = [self._get_default_description(fname) for fname in file_name]
        elif isinstance(description, str):
            description = [description]
        
        if len(file_name) != len(description):
            raise ValueError("The number of files and descriptions must match.")
        
        for fname, desc in zip(file_name, description):
            self.files.append({
                'file_name': fname,
                'description': desc
            })

    def add_action(self, step_count: int, tool_name: str, sub_goal: str, command: str, result: Any) -> None:
        action = {
            'tool_name': tool_name,
            'sub_goal': sub_goal,
            'command': command,
            'result': result,
        }
        step_name = f"Action_Step_{step_count}"
        self.actions[step_name] = action

    def get_query(self) -> Optional[str]:
        return self.query

    def get_files(self) -> List[Dict[str, str]]:
        return self.files
    
    # def get_actions(self) -> Dict[str, Dict[str, Any]]:
    #     return self.actions
    