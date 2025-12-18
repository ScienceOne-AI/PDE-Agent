##############################################################
# Jermain
# main mechanism for PDE Agent。
##############################################################


import os
import json
import random
from tqdm import tqdm
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.getcwd())

from .toolkits.deepxde import XDE_Toolkit
from .minds.mind import Mind
from .rag.rag import PDE_RAG
from .minds import Mind_Ensemble
from .minds.executor import Executor
from .utils.tool_chain import Tool_Chain_Analyzer
from .memory import Resources_Pool
from .utils import save_checkpoint, update_step_info, adjust_execution_file



class Hive_Mind:
    def __init__(
        self, 
        query, 
        model_info='deepseek-chat', 
        root_cache_dir=None, 
        verbose: bool = True,
        ):
        self.query = query
        self.model_info = model_info
        self.mind_ensemble = self.initialize_mind_ensemble(model_info=model_info)
        self.xde_toolkit = XDE_Toolkit()
        self.resources_pool = Resources_Pool()
        # self.tool_chain = Tool_Chain_Analyzer(self.xde_toolkit.get_tool_names())
        self.tool_chain = Tool_Chain_Analyzer()
        self.token_count = 0
        
    def initialize_assets(self):
        self.resources_pool.clear()
        self.tool_chain.clear()
        
    def pde_solver(self, image_path=None, loop_num=2):
        # RAG case
        logging.info("1. 🔍 RAG case")

        embedding_model_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../pyenv/models/MathBERT")
        # r"../pyenv/models/MathBERT"
        pde_rag = PDE_RAG(
            embedding_model_name=embedding_model_name,
            top_k=2,
        )
        retrieval_pde_case = pde_rag(self.query)
        # RAG case
        logging.info(f"RAG case: \n{retrieval_pde_case}")
        save_checkpoint({"RAG_case": retrieval_pde_case})
        
        ##################################################
        # planning agent
        self.planner = self.mind_ensemble.initialize_planner(self.xde_toolkit.get_tool_info_dict())
        self.executor = Executor(
            role='Executor', instruction=None, 
            toolkits=self.xde_toolkit.get_tool_dict(),
            model_info=self.model_info,)
        
        # start to solve
        loop_count = 0
        while loop_count < loop_num:
            loop_count += 1

            validation_flag = True
            # TODO: 1 解析任务
            logging.info("2. ⚙️ Subtask analysis")
            try:
                if loop_count == 1:
                    subtask_anylysis = self.planner.analyze_query(self.query, retrieval_pde_case)
                else:
                    adjust_execution_file(loop_count-1)
                    previous_actions = self.orchestrator.share_memory_all() if hasattr(self, 'orchestrator') else []
                    # validation_info = current_validation_info if current_validation_info else subtask_anylysis
                    if 'current_validation_info' in locals():
                        validation_info = current_validation_info if current_validation_info else subtask_anylysis
                    else:
                        validation_info = subtask_anylysis
                    self.initialize_assets()
                    subtask_anylysis = self.planner.analyze_query(self.query, retrieval_pde_case, 
                                                                previous_actions=previous_actions, 
                                                                validation_info=validation_info)
            except Exception as e:
                error_msg = f"Error in subtask analysis: {type(e).__name__} - {e}"
                subtask_anylysis = {"error_msg": error_msg}
                logging.error(error_msg)
                validation_flag = False
            
            # save subtask_anylysis to file
            save_checkpoint({"planning_task": subtask_anylysis,})
            logging.info(f"Subtask analysis: \n{subtask_anylysis}")
            if not validation_flag:
                continue
        
        
            # TODO: 2 工具调用
            logging.info("3. ⚒️ Tool use")
            # multi-agent
            logging.info("ensemble initialization ⏳")
            self.orchestrator = self.mind_ensemble.initialize_orchestrator(subtask_anylysis, retrieval_pde_case)
            self.parser = self.mind_ensemble.initialize_parser()
            self.solver = self.mind_ensemble.initialize_solver()
            # 工具调用
            previous_actions = []
            # valid_actions = []
            active_help_flag = False
            parser_tool_list = self.xde_toolkit.get_parser_tool_names()
            for idx, task_info in enumerate(subtask_anylysis):
                tool_name = task_info['tool_name']
                task_desc = task_info['reasoning']
                tool_info = [item for item in self.xde_toolkit.tool_info.items() if item[0] == tool_name]
                tool_info = tool_info[0] if tool_info else None
                
                if not tool_info:
                    # raise ValueError(f"Invalid tool name: {tool_name}")
                    logging.warning(f"Invalid tool name: {tool_name}")
                self.tool_chain.add_tool_node(tool_name)
                # logging
                logging.info(f"Step {idx+1}: {tool_name} - {task_desc}")
                
                
                try:
                    if tool_name in parser_tool_list:
                        implementor = self.parser
                    else:
                        implementor = self.solver
                    previous_actions = self.orchestrator.share_memory_all()
                    tool_params = implementor.process_tool_use(tool_info, previous_actions)
                    tool_params_kwargs, conn_action_step = self.resources_pool.unpack_tool_params(tool_params, tool_name)
                    self.tool_chain.add_current_connection(tool_name, conn_action_step)
                except Exception as e:
                    error_msg = f"Error in processing tool use: {type(e).__name__} - {e}"
                    tool_params_kwargs = {"error_msg": error_msg}
                    active_help_flag = True

                try:
                    # TODO: 3 tool implement
                    tool_result = self.executor.execute_tool(tool_name, tool_params_kwargs)
                    self.resources_pool.add_resource(idx+1, tool_result)
                except Exception as e:
                    error_msg = f"Error in executing tool: {type(e).__name__} - {e}"
                    tool_result = {"error_msg": error_msg}
                    active_help_flag = True
                
                # TODO: 4 解析结果
                current_action = {
                    "step": idx+1,
                    "agent": implementor.role, 
                    "tool_name": tool_name,
                    "params": tool_params_kwargs,
                    "return": {**tool_result},
                }
                self.orchestrator.memory.add_memory_info({tool_name: current_action.copy()})
                
                # save checkpoint
                current_action.update({"loop_count": loop_count})
                save_checkpoint({"execute_tool_chain": update_step_info(current_action, tool_params), })
                # logging
                logging.info(f"Step_{idx+1} - {tool_name} current_action: \n{current_action}")
                
                # TODO: 5 validation
                # if active_help_flag:
                if active_help_flag or subtask_anylysis[min(idx+1, len(subtask_anylysis)-1)]['tool_name'] in ['create_training_data'] or tool_name in ['visualize_and_save']:
                    turn_idx = 0
                    current_validation_info = self.orchestrator.analyze_validate_process(self.query)
                    validation_flag = False if current_validation_info['validation'].lower() == 'invalid' else True

                    current_validation_info_save = current_validation_info.copy()
                    current_validation_info_save.update({"loop_count": loop_count, "turn_idx": turn_idx+1, })
                    save_checkpoint({"valid_steps_info": current_validation_info_save})
                    # logging
                    logging.info(f"Step_{idx+1} - current_validation_info: \n{current_validation_info_save}")
                    
                    while not validation_flag and turn_idx < 2:
                        turn_idx += 1

                        re_run = current_validation_info['action']['re_run']    # list
                        false_action_tool = [self.tool_chain.step_to_tool_name(step) for step in re_run]
                        # truncate tool chain and fix memory
                        repeat_subtask_tool, _ = self.tool_chain.truncate(re_run, show=False)
                        misleading_actions = self.orchestrator.memory.eliminate_false_memory(repeat_subtask_tool)
                        self.resources_pool.eliminate_false_resources([self.tool_chain.tool_name_to_step(tool_name) for tool_name in repeat_subtask_tool])
                        # file save refresh
                        adjust_execution_file(loop_count, turn_idx)
                        
                        self.retool(repeat_subtask_tool, current_validation_info, 
                                    parser_tool_list, false_action_tool, misleading_actions, 
                                    loop_count, turn_idx)
                            
                        # validation
                        current_validation_info = self.orchestrator.analyze_validate_process(self.query)

                        current_validation_info_save = current_validation_info.copy()
                        current_validation_info_save.update({"loop_count": loop_count, "turn_idx": turn_idx+1, })
                        save_checkpoint({"valid_steps_info": current_validation_info_save})
                        # logging
                        logging.info(f"Step_{idx+1}_{turn_idx+1} - current_validation_info: \n{current_validation_info_save}")
                        
                        validation_flag = False if current_validation_info['validation'].lower() == 'invalid' else True
                    
                    active_help_flag = False
                
                # Domestic circulation is false, need to re-solve the task
                # if not validation_flag inner_loop_count, break, and replan
                if not validation_flag:
                    # loop_count += 1
                    break
            
            if validation_flag:
                break           
        # TODO: 7 Task solved or not
        if not validation_flag:
            logging.error(f"\n🤔 Didn't find a solution for the task.\n")
            pass_flag = False
        else:
            # task solved
            logging.info("\n🎉 Finished task execution.\n")
            pass_flag = True
            
        # 输出总的 token 数
        save_checkpoint({"execute_tool_chain": {"pass_flag": pass_flag}})
        # logging
        logging.info(f"Pass flag: {pass_flag}")
        
    def initialize_mind_ensemble(self, model_info='deepseek-chat'):
        inst_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'context')
        return Mind_Ensemble(user_input=self.query, inst_root=inst_root, model_info=model_info)
    
    def retool(self, 
               repeat_subtask_tool, 
               current_validation_info, 
               parser_tool_list, 
               false_action_tool, misleading_actions,
               loop_count, turn_idx):
        for tool_name in repeat_subtask_tool:
            # tool_info = [item for item in self.xde_toolkit.tool_info.items() if item[0] == tool_name]
            tool_info = [item for item in self.xde_toolkit.tool_info.items() if item[0] == tool_name]
            tool_info = tool_info[0] if tool_info else None
            try:
                if tool_name in parser_tool_list:
                    implementor = self.parser
                else:
                    implementor = self.solver
                # tool_params_kwargs, conn_action_step = self.parser.process_tool_use(tool_info, previous_actions)
                # self.tool_chain.add_edges_from(conn_action_step)
                previous_actions = self.orchestrator.share_memory_all()
                
                # false_action,  action_item
                if tool_name in false_action_tool:
                    false_action = misleading_actions[tool_name]
                    action_item = current_validation_info['action'].get('action_items')[false_action_tool.index(tool_name)]
                tool_params = implementor.process_tool_use(tool_info, previous_actions, false_action, action_item)
                tool_params_kwargs, conn_action_step = self.resources_pool.unpack_tool_params(tool_params, tool_name)
                self.tool_chain.add_current_connection(tool_name, conn_action_step)

            except Exception as e:
                error_msg = f"Error in processing tool use: {type(e).__name__} - {e}"
                tool_params_kwargs = {"error_msg": error_msg}
                break
                # active_help_flag = True
            try:
                tool_result = self.executor.execute_tool(tool_name, tool_params_kwargs)
                self.resources_pool.add_resource(self.tool_chain.tool_name_to_step(tool_name), tool_result)
            except Exception as e:
                error_msg = f"Error in executing tool: {type(e).__name__} - {e}"
                tool_result = {"error_msg": error_msg}
                break
                # active_help_flag = True
            
            # TODO: refresh memory
            current_action = {
                # "step": idx+1,
                "step": self.tool_chain.tool_name_to_step(tool_name),
                "agent": implementor.role, 
                "tool_name": tool_name,
                "params": tool_params_kwargs,
                "return": {**tool_result},
            }
            self.orchestrator.memory.add_memory_info({tool_name: current_action.copy()})
            current_action.update({ "loop_count": loop_count, "turn_idx": turn_idx+1})
            save_checkpoint({"execute_tool_chain": update_step_info(current_action, tool_params)})
            # logging
            logging.info(f"{turn_idx+1} - {tool_name} current_action: \n{current_action}")


    
    