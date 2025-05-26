from typing import Optional, List, Mapping, Any
from termcolor import colored
import time
from typing import Optional

from toolbench.utils import process_system_message
from toolbench.inference.utils import SimpleChatIO, generate_stream, react_parser
from openai import OpenAI,OpenAIError
class LlamaModel:
    def __init__(self,key, model_name_or_path,base_url,  template:str="tool-llama-single-round", device: str="cuda", cpu_offloading: bool=False, max_sequence_length: int=2048) -> None:
        super().__init__()
        self.model_name = model_name_or_path
        self.template = template
        self.max_sequence_length = max_sequence_length
        self.client = OpenAI(base_url=base_url, api_key=key) if base_url else OpenAI(api_key=key)

    def prediction(self, prompt: str) -> str:
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(**prompt)
                prediction = response.choices[0].message.content.strip()
                total_tokens = response.usage.total_tokens

                return prediction, total_tokens
            except OpenAIError as e:
                if attempt < max_retries:
                    time.sleep(100)
                else:
                    return str(e), 0

    def add_message(self, message):
        self.conversation_history.append(message)

    def change_messages(self,messages):
        self.conversation_history = messages

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        print("before_print"+"*"*50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            print(print_obj)
            if "function_call" in message.keys():
                print_obj = print_obj + f"function_call: {message['function_call']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)

    def parse(self,functions,process_id,**args):

        self.time = time.time()
        conversation_history = self.conversation_history
        # prompt = ''
        prompt = {
            "model": self.model_name,
            "max_tokens": 1024,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            **args
        }
        messages=[]


        for message in conversation_history:
            content = message['content']
            if message['role'] == "system" and functions != []:
                content = process_system_message(content, functions)
            messages.append({"role": message['role'], "content": content})
        prompt.update({"messages": messages})


        predictions, decoded_token_len = self.prediction(prompt)

        if process_id == 0:
            print(f"[process({process_id})]total tokens: {decoded_token_len}")

        thought, action, action_input = react_parser(predictions)

        message = {
            "role": "assistant",
            "content": predictions,
            "function_call": {
                "name": action,
                "arguments": action_input
            }
        }

        return message, 0, decoded_token_len
