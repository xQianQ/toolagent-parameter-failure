#!/usr/bin/env python
# coding=utf-8
import time
from termcolor import colored
from typing import Optional, List
import torch
from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from toolbench.utils import process_system_message
from toolbench.model.model_adapter import get_conversation_template
from toolbench.inference.utils import SimpleChatIO, generate_stream, react_parser
import json, string, random


class ToolLLaMA:
    def __init__(
            self, 
            model_name_or_path: str, 
            template:str="tool-llama-single-round", 
            device: str="cpu",
            max_sequence_length: int=8192
        ) -> None:
        super().__init__()
        self.model_name = model_name_or_path
        self.template = template
        self.max_sequence_length = max_sequence_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, model_max_length=self.max_sequence_length)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, low_cpu_mem_usage=True, device_map='auto'
        )

        if self.tokenizer.pad_token_id == None:
            self.tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.use_gpu = (True if device.startswith("cuda") else False)
        self.chatio = SimpleChatIO()

    def prediction(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        with torch.no_grad():
            gen_params = {
                "model": "",
                "prompt": prompt,
                "temperature": 0,
                "max_new_tokens": 1024,
                "stop": "</s>",
                "stop_token_ids": None,
                "echo": False
            }
            generate_stream_func = generate_stream
            output_stream = generate_stream_func(self.model, self.tokenizer, gen_params, "cuda", self.max_sequence_length, force_generate=True)
            outputs = self.chatio.return_output(output_stream)
            prediction = outputs.strip()
        return prediction
        
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
            # if "function_call" in message.keys():
            #     print_obj = print_obj + f"function_call: {message['function_call']}"
            if "function_call" in message.keys():
                print_obj = print_obj + f"function_call: {message['function_call']}"
            if 'tool_calls' in message.keys():
                print_obj = print_obj + f"tool_calls: {message['tool_calls']}"
                print_obj = print_obj + f"number of tool calls: {len(message['tool_calls'])}"
            if detailed:
                print_obj = print_obj + f"function_call: {message['function_call']}"
                print_obj = print_obj + f"tool_calls: {message['tool_calls']}"
                print_obj = print_obj + f"function_call_id: {message['function_call_id']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)

    def parse(self, tools, process_id, **args):
        conv = get_conversation_template(self.template)
        if self.template == "tool-llama":
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        elif self.template == "tool-llama-single-round" or self.template == "tool-llama-multi-rounds":
            roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "tool": conv.roles[2], "assistant": conv.roles[3]}

        self.time = time.time()
        conversation_history = self.conversation_history

        if tools != []:
            functions = [tool['function'] for tool in tools]


        prompt = ''
        for message in conversation_history:
            role = roles[message['role']]
            content = message['content']
            if role == "System" and tools != []:
                content = process_system_message(content, functions=functions)
            prompt += f"{role}: {content}\n"
        prompt += "Assistant:\n"
        if tools != []:
            predictions = self.prediction(prompt)
        else:
            predictions = self.prediction(prompt)

        decoded_token_len = len(self.tokenizer(predictions))
        if process_id == 0:
            print(f"[process({process_id})]total tokens: {decoded_token_len}")
        thought, action, action_input = react_parser(predictions)
        random_id = ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(8)])
        message = {
            "role": "assistant",
            "content": predictions,
            "tool_calls": [{
                'id': f"call_{random_id}",
                'type': "function",
                'function': {
                    'name': action,
                    'arguments': action_input
                }
            }]
        }
        return message, 0, decoded_token_len

