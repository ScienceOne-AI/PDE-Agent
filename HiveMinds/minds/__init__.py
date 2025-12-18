import os
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from . import planner, orchestrator, parser, solver

class Mind_Ensemble:
    # inst_root 
    def __init__(self, user_input: str, inst_root: str, model_info: str = 'deepseek-chat'):
        self.user_input = user_input
        self.inst_root = inst_root
        self.model_info = model_info
        with open(os.path.join(inst_root, "instruction.yaml"), "r", encoding="utf-8") as f:
            self.instruction = yaml.safe_load(f)
            
    def initialize_planner(self, toolkit_info: Dict = None):
        planner_inst = self.instruction.get('thinking_prompt', '').strip()
        return planner.Planner(
            role='Planner agent', 
            instruction=planner_inst, 
            toolkit_info=toolkit_info,
            model_info=self.model_info, 
            is_recorded=False, 
            is_continue=False)
        
    def initialize_orchestrator(self, subtask_info, retrieval_case=None):
        if retrieval_case:
            rag_case_prompt = self.instruction.get('rag_case_prompt', '').strip().format(
                RAG_case_problem=retrieval_case["problem_description"], 
                RAG_case_solution=''.join('\n' + json.dumps(retrieval_case["expected_tool_chain"], ensure_ascii=False, indent=4) + '\n'))
        else:
            rag_case_prompt = ''
        pde_orchestrator_prompt = self.instruction.get('pde_orchestrator_prompt_opt').strip().format(
            user_input=self.user_input, 
            task_split_procedure=subtask_info, 
            RAG_case_prompt=rag_case_prompt)
        return orchestrator.Orchestrator(
            role='PDE Orchestrator', 
            instruction=pde_orchestrator_prompt, 
            model_info=self.model_info, 
            is_recorded=False)
        
    def initialize_parser(self):
        parser_inst = self.instruction.get('pde_parser_prompt', '').strip().format(user_input=self.user_input)
        return parser.Parser(
            role='PDE Parser', 
            instruction=parser_inst, 
            model_info=self.model_info, 
            is_recorded=False)
        
    def initialize_solver(self):
        solver_inst = self.instruction.get('pde_solver_prompt', '').strip().format(user_input=self.user_input)
        return solver.Solver(
            role='PDE Solver', 
            instruction=solver_inst, 
            model_info=self.model_info, 
            is_recorded=False)
        
        
    



    