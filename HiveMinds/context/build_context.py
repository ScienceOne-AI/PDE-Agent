import os
import json


def build_difficulty_prompt(question):
    return f"""
Given the following differential equation:  
{question}  

Please evaluate the difficulty of solving it with DeepXDE based on these criteria:  

1) Easy:  
    - The equation is linear, low-dimensional (1D/2D), and has standard boundary conditions.  
    - Can be solved directly using default DeepXDE configurations without additional tuning.  

2) Medium:  
    - The equation contains nonlinear terms, moderate-dimensional domains (3D), or complex boundary conditions (e.g., time-dependent, discontinuous).  
    - Requires parameter tuning, custom loss functions, or retrieval of similar solutions (RAG) for optimization.  

3) Difficult:  
    - The equation is high-dimensional (4D+), involves multi-physics coupling, or requires adaptive methods (e.g., PINNs with domain decomposition).  
    - Needs collaborative workflows (e.g., multi-model integration, hybrid numerical-PINN approaches) or advanced techniques.  

Provide the difficulty level [1) Easy, 2) Medium, or 3) Difficult] and a brief justification.
    """
    

    prompt = f"""
You are an intelligent assistant specialized in solving partial differential equations (PDEs) using the DeepXDE library.
I will provide a set of Python functions written with DeepXDE as tools for solving PDE problems.
Your task is to generate a step-by-step procedure for calling these tools to solve the PDE problem based on the user's input and your understanding of the problem.

### Task Requirements:
1. Output in JSON format that can be parsed by Python's json.loads() method
2. Use LaTeX format for mathematical expressions, consistent with the user's input
3. Escape newline (\n) or tab (\t) characters as \\n or \\t in JSON strings
4. Each step must include a clear rationale for tool selection in plain English (no mathematical formulas)

### Available Tool Functions:
{{
    "define_pde": "{tools_info_dict.get('define_pde', None)}",
    "create_network": "{tools_info_dict.get('create_network', None)}",
    "define_reference_solution": "{tools_info_dict.get('define_reference_solution', None)}",
    "train_model_LBFGS": "{tools_info_dict.get('train_model_LBFGS', None)}",
    "define_initial_condition": "{tools_info_dict.get('define_initial_condition', None)}",
    "create_training_data": "{tools_info_dict.get('create_training_data', None)}",
    "define_boundary_condition": "{tools_info_dict.get('define_boundary_condition', None)}",
    "train_model": "{tools_info_dict.get('train_model', None)}",
    "visualize_and_save": "{tools_info_dict.get('visualize_and_save', None)}",
    "define_domain": "{tools_info_dict.get('define_domain', None)}"
}}

### Output Format Specifications:
<think>
The thinking process for the solution process is as follows:
</think>
[
{{"task_id": 1, "tool_name": "Selected tool name", "reasoning": "Reason for selecting the tool"}},
...
]

Please process the following user input:
{user_input}
"""
    return prompt

def build_subtask_prompt_zero(user_input: str, tools_info_dict: dict) -> str:
    prompt = f"""
You are specialized in solving partial differential equations (PDEs) using the DeepXDE library.
I will provide a set of tools functions written with DeepXDE as tools for solving PDE problems.
Your task is to generate a step-by-step procedure for calling these tools to solve the PDE problem based on the user's input and your understanding of the problem.

### Task Requirements:
1. Output in JSON format that can be parsed by Python's json.loads() method
2. Use LaTeX format for mathematical expressions, consistent with the user's input
3. LaTeX equations: for example, use \\\\frac (four backslashes) instead of \\frac (two backslashes or others) in JSON strings to avoid parsing issues
4. Escape newline (\n) or tab (\t) characters as \\n or \\t in JSON strings
5. Each step must include a clear rationale for tool selection

### Available Tool Functions:
{{
    "define_pde": "{tools_info_dict.get('define_pde', None)}",
    "create_network": "{tools_info_dict.get('create_network', None)}",
    "define_reference_solution": "{tools_info_dict.get('define_reference_solution', None)}",
    "train_model_LBFGS": "{tools_info_dict.get('train_model_LBFGS', None)}",
    "define_initial_condition": "{tools_info_dict.get('define_initial_condition', None)}",
    "create_training_data": "{tools_info_dict.get('create_training_data', None)}",
    "define_boundary_condition": "{tools_info_dict.get('define_boundary_condition', None)}",
    "train_model": "{tools_info_dict.get('train_model', None)}",
    "visualize_and_save": "{tools_info_dict.get('visualize_and_save', None)}",
    "define_domain": "{tools_info_dict.get('define_domain', None)}"
}}

### Output Format Specifications (thinking process included in <think> and </think> tags, and JSON output):
<think>
The thinking process for the solution process.
</think>
[
{{"task_id": 1, "tool_name": "Selected tool name", "reasoning": "Reason for selecting the tool"}},
...
]

Please process the following user input and ensure that the output can be parsed by Python's json.loads() method:
{user_input}
"""
    return prompt


def build_retrieval_prompt(user_input: str) -> str:
    PROMPT_Find: str = """
Find the PDE problem case that most closely matches the following case:
{user_input}

Priority: match the specific PDE equation type first if provided.
"""
    prompt = PROMPT_Find.format(user_input=user_input)
    return prompt


def build_tool_use_prompt(tool_info, previous_steps_info):
    # 得到工具的参数信息
    tool_params_prompt = "\n".join(
        f"Params {i + 1}:[\n\tname: {key}\n\ttype: {value['type']}\n\tdesc: {value['description']}\n]"
        for i, (key, value) in enumerate(tool_info[1]["params"].items())
    )
    prompt = f"""
### Tool info:
Name: {tool_info[0]}
Description: {tool_info[1]["desc"]}
Required params: 
{tool_params_prompt}

### Previous tool exec results (includes tool name, params, result type & value):
{previous_steps_info}

Please provide parameter values for the tool. Output a JSON object parsable by json.loads().

Instructions:
- First, detail your thinking process enclosed in <think> and </think> tags.
- Then, provide the JSON answer on a new line.
"""
    return prompt


def build_tool_use_prompt_false(tool_info, previous_steps_info, current_step_info, action_item):

    tool_params_prompt = build_tool_use_prompt(tool_info, previous_steps_info)
    prompt = f"""
Current step & tool ({current_step_info['tool_name']}) context: agent, tool, params, result (type/value/error)
{current_step_info}

PDE Orchestrator validation found issues with tool parameters/results:
{action_item}

Re-evaluate using this reasoning and provide revised parameters:
{tool_params_prompt}
"""
    return prompt


def build_subtask_prompt(question: str, retrieval_case: str, tools_info_dict: dict) -> str:
    init_prompt = build_subtask_prompt_zero(question, tools_info_dict)
    # retrieval_case 是一个字典
    case_example = """
### Example:
#### User input
{retrieval_problem}

#### output
<think>
The thinking process for the solution process is as follows:
</think>
[
{task_plan}
]
"""
    # 构建 task_plan
    task_plan = ""
    for idx, task in enumerate(retrieval_case['expected_tool_chain']):
        if idx < 2:
            task_plan += f"""{{"task_id": {int(task['step'])}, "tool_name": "{task['tool_name']}", "reasoning": "provide the reasoning brifly"}},\n"""
        else:
            task_plan += f"""{{"task_id": {int(task['step'])}, "tool_name": "{task['tool_name']}", "reasoning": ""}},\n"""
    task_plan = task_plan.strip(",\n")
    
    retrieval_case_plan = case_example.format(retrieval_problem=retrieval_case['problem_description'], task_plan=task_plan)
    
    PROMPT_Add_Retrieval_Case: str = """
{prompt}

We found the following case for reference and Note: the tool, define_reference_solution, is needed when a reference solution is specified:
{retrieval_case_plan}
"""
    return PROMPT_Add_Retrieval_Case.format(prompt=init_prompt, retrieval_case_plan=retrieval_case_plan)


def build_validation_prompt(previous_steps_info, user_input: str):
    # test_previous_steps_info = previous_steps_info.copy()
    # test_previous_steps_info.append(current_step_info)
    prompt = f"""
The PDE problem is: 
{user_input}

At present, the step & tool ({[item['tool_name'] for item in previous_steps_info]}) execution completed. Verify the reliability of the parameters and results in relation to the input?? [valid/invalid]
- Context of preceding steps: for each step: agent, tool name, parameters, result (type, value, or error_msg), where the presence of error_msg indicates an execution failure.
{previous_steps_info}
"""
    return prompt


def build_re_subtask_prompt(question: str, retrieval_case: str, tools_info_dict: dict, 
                         task_split_info, previous_steps_info, validation_info) -> str:
    task_split_prompt = build_subtask_prompt(question, retrieval_case, tools_info_dict)
    prompt = f"""
The current task split is:
{task_split_info}
But PDE Orchestrator validation found issues when executing, the excutable tool chain is (include tool name, parameters, result type/value/error):
{previous_steps_info}
and the error context is:
{validation_info}

{task_split_prompt}
"""
    return prompt

    
if __name__ == '__main__':
    print(build_difficulty_prompt("test_problem"))
    
# Please note:
#   - You should first provide a detailed explanation of your thinking process, using the start tag <think> and end tag </think> to identify the thinking process;
#   - Followed the end tag </think>, provide the specific answer content and pay attention to the line breaks.