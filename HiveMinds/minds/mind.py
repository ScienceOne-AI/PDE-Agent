import os
import re
from HiveMinds.engine.factory import create_llm_engine
from HiveMinds.memory.memory import Memory

class Mind:
    def __init__(self, role, instruction, model_info=None, examplers=None, img_path=None, is_recorded=False, is_continue=False):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info if model_info else os.getenv('CHAT_MODEL', 'gpt-4o-mini')
        self.img_path = img_path
        self.is_recorded = is_recorded
        self.is_continue = is_continue
        self.current_token = {}
        self.token_count = {}
        self.mimd = create_llm_engine(model_info=self.model_info, is_multimodal=False)
        self.memory = Memory()
        self.messages = [
                {"role": "system", "content": instruction},
            ]
        if examplers is not None:
            for exampler in examplers:
                self.messages.append({"role": "user", "content": exampler['question']})
                self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})
        if self.is_recorded:
            self.record_path = os.path.join(os.getenv('OUTPUT_DIR'), f'{self.role}_record.txt')
            with open(self.record_path, 'w', encoding='utf-8') as f:
                for item in self.messages:
                    f.write("#"*5 + item['role'] + ':'+ '\n' + item['content'] + '\n\n')
        ##################################################################
        
    def set_instruction(self, instruction):
        self.instruction = instruction
        self.messages[0]['content'] = instruction

    def chat(self, message, instruction=None, img_path=None):
        if instruction is not None:
            self.set_instruction(instruction)
        if self.is_continue:
            self.messages.append({"role": "user", "content": message})
            current_messages = self.messages.copy()
        else:
            current_messages = self.messages.copy()
            current_messages.append({"role": "user", "content": message})
            
        response = self.mimd(current_messages)

        think_content, answer_content = self.response_split(response)
        answer_content = answer_content.strip().replace("json", "").replace("```", '')
        self.current_token = self.mimd.current_token
        self.token_count.update({
                "input": current_messages,
                "output_think": think_content,
                "output_answer": answer_content,
                "current_token": self.current_token,
            })

        if self.is_recorded:
            with open(self.record_path, 'a', encoding='utf-8') as f:
                f.write("user: " + '\n' + message + '\n\n')
                f.write("assistant: " + '\n' + response.choices[0].message.content + '\n\n')
                f.write("token_count: " + '\n' + str(self.token_count) + '\n\n')
        if self.is_continue:
            self.messages.append({"role": "assistant", "content": answer_content})
            
        # TODO: Debug
        # if self.role == 'PDE Solver':
        #     # 工具解析
        #     tool_params = json.loads(answer_content)
        #     if tool_params.get('is_plot', False):
        #         param_from_step = tool_params.get('loss_history', None)
        #         print(param_from_step['value'])
        #         if param_from_step.startswith('Step_7'):
        #             current_messages.append({"role": "assistant", "content": response.choices[0].message.content})
        #             current_messages.append({"role": "user", "content": "why you choose the parameters from step 7? why not step 8? is it because the world given has problem?"})
        # ##############################################################################################
            
        return think_content, answer_content
        
        
    def response_split(self, content: str) -> tuple:
        match = re.search(r'<think>(.*?)</think>(.*)', content, re.DOTALL)
        if match:
            think_content = match.group(1).strip()
            answer_content = match.group(2).strip().replace("json", "").replace("```", '')
            return think_content, answer_content
        else:
            return "No valid <think> tag found", content
        
    def get_current_token(self):
        return self.current_token
    
    def get_token_count(self):
        return self.token_count
    
    def add_to_memory(self, memory):
        pass