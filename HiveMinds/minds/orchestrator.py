import os
import re
from PIL import Image
from typing import Dict, Any, List, Tuple, Union, Optional
import json
import logging

# from HiveMinds.memory import Memory
from HiveMinds.minds.mind import Mind
from HiveMinds.context.build_context import build_validation_prompt

# from octotools.engine.factory import create_llm_engine
# from octotools.models.memory import Memory
# from octotools.models.formatters import QueryAnalysis, NextStep, MemoryVerification

class Orchestrator(Mind):
    ROLE = "Orchestrator"
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

    def analyze_validate_process(self, 
                                 question: str):
        validation_prompt = build_validation_prompt(self.memory.get_memory_values(), question)
        
        validation_think, validation_answer = self.chat(validation_prompt)
        validation_info = json.loads(validation_answer)
        
        # process validation_info
        re_run = validation_info['action'].get('re_run', [])
        if re_run:
            if isinstance(re_run, (str, int)):
                validation_info['action']['re_run'] = [int(re_run)]
            elif isinstance(re_run, list):
                validation_info['action']['re_run'] = [int(i) for i in re_run]
        # process validation_info  [[action_item1], [action_item2]]
        action_items = validation_info['action'].get('action_items', [])
        action_items = self._adjust_action_items(action_items, validation_info['action']['re_run'])
        validation_info['action']['action_items'] = action_items
        
        # # FIXME: This is a temporary, need to refactor this part
        # if not previous_subtasks:
        #     subtask_prompt = subtask_prompt_with_rag(question, retrieval_case, self.toolkit_info)
        # else:
        #     # FIXME
        #     # adjust_execution_file(loop_count-1) # 这是调整保存的文件，还需要修改
        #     # 下面变量还需要修改
        #     subtask_prompt = re_subtask_prompt(question, retrieval_case, self.toolkit_info,
        #                                             task_split_info, previous_actions, error_step)


        # logging
        logging.info(f"Validation Prompt: \n{validation_prompt}")
        logging.info(f"Validation think: \n{validation_think}")
        logging.info(f"Validation answer: \n{validation_answer}")
        logging.info(f"Validation info: \n{validation_info}")
        
        
        return validation_info
    
    def _adjust_action_items(self, action_items, re_run):
        action_items_len = len(action_items)
        re_run_len = len(re_run)

        if action_items_len == re_run_len:
            return action_items
        elif action_items_len < re_run_len:
            # 直接拼接：原b的切片 + 空字符串补足部分（用*生成，高效）
            return action_items[:] + [''] * (re_run_len - action_items_len)
        else:
            # 列表推导式生成len(a)个b的副本，比for循环append快
            return [action_items[:] for _ in range(re_run_len)]
    
    def share_memory_all(self) -> list:
        return list(self.memory.memory_info.values())

#     def extract_context_subgoal_and_tool(self, response: Any) -> Tuple[str, str, str]:

#         def normalize_tool_name(tool_name: str) -> str:
#             # Normalize the tool name to match the available tools
#             for tool in self.available_tools:
#                 if tool.lower() in tool_name.lower():
#                     return tool
#             return "No matched tool given: " + tool_name
        
#         try:
#             if isinstance(response, NextStep):
#                 context = response.context.strip()
#                 sub_goal = response.sub_goal.strip()
#                 tool_name = response.tool_name.strip()
#             else:
#                 text = response.replace("**", "")

#                 # Pattern to match the exact format
#                 pattern = r"Context:\s*(.*?)Sub-Goal:\s*(.*?)Tool Name:\s*(.*?)(?=\n\n|\Z)"
                
#                 # Find all matches
#                 matches = re.findall(pattern, text, re.DOTALL)

#                 # Return the last match (most recent/relevant)
#                 context, sub_goal, tool_name = matches[-1]
#                 context = context.strip()
#                 sub_goal = sub_goal.strip()
#             tool_name = normalize_tool_name(tool_name)
#         except Exception as e:
#             print(f"Error extracting context, sub-goal, and tool name: {str(e)}")
#             return None, None, None

#         return context, sub_goal, tool_name
        
#     def generate_next_step(self, question: str, image: str, query_analysis: str, memory: Memory, step_count: int, max_step_count: int) -> Any:
#         prompt_generate_next_step = f"""
# Task: Determine the optimal next step to address the given query based on the provided analysis, available tools, and previous steps taken.

# Context:
# Query: {question}
# Image: {image}
# Query Analysis: {query_analysis}

# Available Tools:
# {self.available_tools}

# Tool Metadata:
# {self.toolbox_metadata}

# Previous Steps and Their Results:
# {memory.get_actions()}

# Current Step: {step_count} in {max_step_count} steps
# Remaining Steps: {max_step_count - step_count}

# Instructions:
# 1. Analyze the context thoroughly, including the query, its analysis, any image, available tools and their metadata, and previous steps taken.

# 2. Determine the most appropriate next step by considering:
#    - Key objectives from the query analysis
#    - Capabilities of available tools
#    - Logical progression of problem-solving
#    - Outcomes from previous steps
#    - Current step count and remaining steps

# 3. Select ONE tool best suited for the next step, keeping in mind the limited number of remaining steps.

# 4. Formulate a specific, achievable sub-goal for the selected tool that maximizes progress towards answering the query.

# Response Format:
# Your response MUST follow this structure:
# 1. Justification: Explain your choice in detail.
# 2. Context, Sub-Goal, and Tool: Present the context, sub-goal, and the selected tool ONCE with the following format:

# Context: <context>
# Sub-Goal: <sub_goal>
# Tool Name: <tool_name>

# Where:
# - <context> MUST include ALL necessary information for the tool to function, structured as follows:
#   * Relevant data from previous steps
#   * File names or paths created or used in previous steps (list EACH ONE individually)
#   * Variable names and their values from previous steps' results
#   * Any other context-specific information required by the tool
# - <sub_goal> is a specific, achievable objective for the tool, based on its metadata and previous outcomes.
# It MUST contain any involved data, file names, and variables from Previous Steps and Their Results that the tool can act upon.
# - <tool_name> MUST be the exact name of a tool from the available tools list.

# Rules:
# - Select only ONE tool for this step.
# - The sub-goal MUST directly address the query and be achievable by the selected tool.
# - The Context section MUST include ALL necessary information for the tool to function, including ALL relevant file paths, data, and variables from previous steps.
# - The tool name MUST exactly match one from the available tools list: {self.available_tools}.
# - Avoid redundancy by considering previous steps and building on prior results.
# - Your response MUST conclude with the Context, Sub-Goal, and Tool Name sections IN THIS ORDER, presented ONLY ONCE.
# - Include NO content after these three sections.

# Example (do not copy, use only as reference):
# Justification: [Your detailed explanation here]
# Context: Image path: "example/image.jpg", Previous detection results: [list of objects]
# Sub-Goal: Detect and count the number of specific objects in the image "example/image.jpg"
# Tool Name: Object_Detector_Tool

# Remember: Your response MUST end with the Context, Sub-Goal, and Tool Name sections, with NO additional content afterwards.
# """
#         next_step = self.llm_engine(prompt_generate_next_step, response_format=NextStep)
#         self.current_token = self.llm_engine.get_current_tokens()
        
#         return next_step

#     def verificate_context(self, question: str, image: str, query_analysis: str, memory: Memory) -> Any:
#         image_info = self.get_image_info(image)

#         prompt_memory_verification = f"""
# Task: Thoroughly evaluate the completeness and accuracy of the memory for fulfilling the given query, considering the potential need for additional tool usage.

# Context:
# Query: {question}
# Image: {image_info}
# Available Tools: {self.available_tools}
# Toolbox Metadata: {self.toolbox_metadata}
# Initial Analysis: {query_analysis}
# Memory (tools used and results): {memory.get_actions()}

# Detailed Instructions:
# 1. Carefully analyze the query, initial analysis, and image (if provided):
#    - Identify the main objectives of the query.
#    - Note any specific requirements or constraints mentioned.
#    - If an image is provided, consider its relevance and what information it contributes.

# 2. Review the available tools and their metadata:
#    - Understand the capabilities and limitations and best practices of each tool.
#    - Consider how each tool might be applicable to the query.

# 3. Examine the memory content in detail:
#    - Review each tool used and its execution results.
#    - Assess how well each tool's output contributes to answering the query.

# 4. Critical Evaluation (address each point explicitly):
#    a) Completeness: Does the memory fully address all aspects of the query?
#       - Identify any parts of the query that remain unanswered.
#       - Consider if all relevant information has been extracted from the image (if applicable).

#    b) Unused Tools: Are there any unused tools that could provide additional relevant information?
#       - Specify which unused tools might be helpful and why.

#    c) Inconsistencies: Are there any contradictions or conflicts in the information provided?
#       - If yes, explain the inconsistencies and suggest how they might be resolved.

#    d) Verification Needs: Is there any information that requires further verification due to tool limitations?
#       - Identify specific pieces of information that need verification and explain why.

#    e) Ambiguities: Are there any unclear or ambiguous results that could be clarified by using another tool?
#       - Point out specific ambiguities and suggest which tools could help clarify them.

# 5. Final Determination:
#    Based on your thorough analysis, decide if the memory is complete and accurate enough to generate the final output, or if additional tool usage is necessary.

# And please note that you must eventually use the visualization tool.

# Response Format:

# If the memory is complete, accurate, AND verified:
# Explanation: 
# <Provide a detailed explanation of why the memory is sufficient. Reference specific information from the memory and explain its relevance to each aspect of the task. Address how each main point of the query has been satisfied.>

# Conclusion: STOP

# If the memory is incomplete, insufficient, or requires further verification, or the visual_save tool is needed in query:
# Explanation: 
# <Explain in detail why the memory is incomplete. Identify specific information gaps or unaddressed aspects of the query. Suggest which additional tools could be used, how they might contribute, and why their input is necessary for a comprehensive response.>

# Conclusion: CONTINUE

# IMPORTANT: Your response MUST end with either 'Conclusion: STOP' or 'Conclusion: CONTINUE' and nothing else. Ensure your explanation thoroughly justifies this conclusion.
# """

#         input_data = [prompt_memory_verification]
#         if image_info:
#             try:
#                 with open(image_info["image_path"], 'rb') as file:
#                     image_bytes = file.read()
#                 input_data.append(image_bytes)
#             except Exception as e:
#                 print(f"Error reading image file: {str(e)}")

#         stop_verification = self.llm_engine_mm(input_data, response_format=MemoryVerification)
#         self.current_token = self.llm_engine_mm.get_current_tokens()
        
#         return stop_verification

#     def extract_conclusion(self, response: Any) -> str:
#         if isinstance(response, MemoryVerification):
#             analysis = response.analysis
#             stop_signal = response.stop_signal
#             if stop_signal:
#                 return analysis, 'STOP'
#             else:
#                 return analysis, 'CONTINUE'
#         else:
#             analysis = response
#             pattern = r'conclusion\**:?\s*\**\s*(\w+)'
#             matches = list(re.finditer(pattern, response, re.IGNORECASE | re.DOTALL))
#             # if match:
#             #     conclusion = match.group(1).upper()
#             #     if conclusion in ['STOP', 'CONTINUE']:
#             #         return conclusion
#             if matches:
#                 conclusion = matches[-1].group(1).upper()
#                 if conclusion in ['STOP', 'CONTINUE']:
#                     return analysis, conclusion
            
#             # If no valid conclusion found, search for STOP or CONTINUE anywhere in the text
#             if 'stop' in response.lower():
#                 return analysis, 'STOP'
#             elif 'continue' in response.lower():
#                 return analysis, 'CONTINUE'
#             else:
#                 print("No valid conclusion (STOP or CONTINUE) found in the response. Continuing...")
#                 return analysis, 'CONTINUE'

#     def generate_final_output(self, question: str, image: str, memory: Memory) -> str:
#         image_info = self.get_image_info(image)

#         prompt_generate_final_output = f"""
# Task: Generate the final output based on the query, image, and tools used in the process.

# Context:
# Query: {question}
# Image: {image_info}
# Actions Taken:
# {memory.get_actions()}

# Instructions:
# 1. Review the query, image, and all actions taken during the process.
# 2. Consider the results obtained from each tool execution.
# 3. Incorporate the relevant information from the memory to generate the step-by-step final output.
# 4. The final output should be consistent and coherent using the results from the tools.

# Output Structure:
# Your response should be well-organized and include the following sections:

# 1. Summary:
#    - Provide a brief overview of the query and the main findings.

# 2. Detailed Analysis:
#    - Break down the process of answering the query step-by-step.
#    - For each step, mention the tool used, its purpose, and the key results obtained.
#    - Explain how each step contributed to addressing the query.

# 3. Key Findings:
#    - List the most important discoveries or insights gained from the analysis.
#    - Highlight any unexpected or particularly interesting results.

# 4. Answer to the Query:
#    - Directly address the original question with a clear and concise answer.
#    - If the query has multiple parts, ensure each part is answered separately.

# 5. Additional Insights (if applicable):
#    - Provide any relevant information or insights that go beyond the direct answer to the query.
#    - Discuss any limitations or areas of uncertainty in the analysis.

# 6. Conclusion:
#    - Summarize the main points and reinforce the answer to the query.
#    - If appropriate, suggest potential next steps or areas for further investigation.
# """

#         input_data = [prompt_generate_final_output]
#         if image_info:
#             try:
#                 with open(image_info["image_path"], 'rb') as file:
#                     image_bytes = file.read()
#                 input_data.append(image_bytes)
#             except Exception as e:
#                 print(f"Error reading image file: {str(e)}")

#         final_output = self.llm_engine_mm(input_data)

#         return final_output


#     def generate_direct_output(self, question: str, image: str, memory: Memory) -> str:
#         image_info = self.get_image_info(image)

#         prompt_generate_final_output = f"""
# Context:
# Query: {question}
# Image: {image_info}
# Initial Analysis:
# {self.query_analysis}
# Actions Taken:
# {memory.get_actions()}

# Please generate the concise output based on the query, image information, initial analysis, and actions taken. Break down the process into clear, logical, and conherent steps. Conclude with a precise and direct answer to the query.

# Answer:
# """

#         input_data = [prompt_generate_final_output]
#         if image_info:
#             try:
#                 with open(image_info["image_path"], 'rb') as file:
#                     image_bytes = file.read()
#                 input_data.append(image_bytes)
#             except Exception as e:
#                 print(f"Error reading image file: {str(e)}")

#         final_output = self.llm_engine_mm(input_data)

#         return final_output
    
    # TODO: jermain get tokens
    def get_current_tokens(self):
        return self.current_token
    