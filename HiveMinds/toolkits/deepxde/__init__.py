"""deepxde toolkit
includes tools for extracting PDE objects and creating nn models
"""
PACKAGE_VERSION = '1.0.0'
__version__ = PACKAGE_VERSION
__author__ = 'jermain'

__all__ = [
    'define_pde',
    'define_reference_solution',
    'define_domain',
    'define_initial_condition',
    'define_boundary_condition',
    'create_training_data',
    'create_network',
    'train_model',
    'train_model_LBFGS',
    'visualize_and_save',
]

from .tools import *
import os
import json

class LazyLoader:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None
    
    def __getattr__(self, name):
        if self.module is None:
            self.module = __import__(f"{__name__}.{self.module_name}", fromlist=[""])
        return getattr(self.module, name)


class XDE_Toolkit:
    def __init__(self):
        # print("XDE_Toolkit")
        self.TOOL_DICT = {
            "define_pde": define_pde,
            "define_reference_solution": define_reference_solution,
            "define_domain": define_domain,
            "define_initial_condition": define_initial_condition,
            "define_boundary_condition": define_boundary_condition,
            "create_training_data": create_training_data,
            "create_network": create_network,
            "train_model": train_model,
            "train_model_LBFGS": train_model_LBFGS,
            "visualize_and_save": visualize_and_save
        }
        self.tool_info = self._load_tool_info()
        
    def _load_tool_info(self):
        tool_info_path = os.path.join(os.path.dirname(__file__), "desc", "tools_info.json")
        with open(tool_info_path, "r", encoding="utf-8") as f:
            tool_info = json.load(f)
        return {tool['name']: {'desc': tool['description'], 'params': tool['parameters']} for tool in tool_info}
        
    def get_tool_info_dict(self):
        return self.tool_info
        
    def get_tool_dict(self):
        return self.TOOL_DICT
    
    def get_parser_tool_names(self):
        return list(self.TOOL_DICT.keys())[:5]
    
    def get_solver_tool_names(self):
        return list(self.TOOL_DICT.keys())[5:]
    
    def get_tool_names(self):
        return list(self.TOOL_DICT.keys())
    

        