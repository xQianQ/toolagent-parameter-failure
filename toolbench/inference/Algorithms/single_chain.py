import json
import re
from Tree.Tree import my_tree, tree_node
from Prompts.ReAct_prompts import FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION, FORMAT_INSTRUCTIONS_USER_FUNCTION
from Algorithms.base_search import base_search_method
from copy import deepcopy
from toolbench.inference.LLM.llama_model import LlamaModel

def fix_brackets(json_str):
    last_brace_index = json_str.rfind('}')
    if last_brace_index != -1:
        json_str = json_str[:last_brace_index + 1]
    stack = []
    for char in json_str:
        if char == '"':
            if stack and stack[-1] == '"':
                stack.pop()
            else:
                stack.append('"')
        if char == '[':
            stack.append(']')
        elif char == '{':
            stack.append('}')
        elif char == ']':
            if stack and stack[-1] == ']':
                stack.pop()
            else:
                json_str += ']'
        elif char == '}':
            if stack and stack[-1] == '}':
                stack.pop()
            else:
                json_str += '}'
    while stack:
        json_str += stack.pop()
    return json_str

def to_camel_case(s):
    words = s.split('_')
    return words[0].lower() + ''.join(word.title() for word in words[1:])


def to_snake_case(s):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def attack_response(observation,attack_type="T1"):
    print("observation：", observation)

    if not observation.endswith('}')and attack_type != "T5":
        observation = fix_brackets(observation)
    try:
        observation_dict = json.loads(eval(json.dumps(observation)))
        if attack_type == "T5":
            response_str = observation_dict["response"]
            response_str = str(response_str)
            if response_str.endswith("}]") or response_str.endswith("}"):
                response_str = response_str[:-2] + "..."

            observation_dict["response"] = response_str

            observation_str = "; ".join([f"{key}:{value}" for key, value in observation_dict.items()])
            return observation_str

        response = observation_dict["response"]
        key_list = []
        if isinstance(response, str):
            if response.startswith('[{'):
                closing_brace_index = response.find('}', 2)
                if closing_brace_index != -1:
                    try:
                        response_temp = eval(response[1:closing_brace_index + 1])
                        key_list.extend(response_temp.keys())
                    except:
                        pass
            if len(key_list) != 0:
                for index, key in enumerate(key_list):
                    if attack_type == "T1":
                        response = response.replace(str(key), "KEY" + str(index))
                    if attack_type == "T2":
                        response = fix_brackets(response)
                        response_list = json.loads(response)
                        for item in response_list:
                            if key == "id" or key == "ID" or "ID" in key.upper():
                                item[key] = "id_" + str(item[key])
                        response = json.dumps(response_list)
                    if attack_type == "T3" and "_" in key:
                        camel_case_key = to_camel_case(key)
                        response = response.replace(str(key), camel_case_key)
                    if attack_type == "T4":
                        snake_case_str = to_snake_case(key)
                        response = response.replace(str(key), snake_case_str)
                    else:
                        print("wrong attack type")

        else:
            if isinstance(response, dict):
                key_list.extend(response.keys())
            for index, key in enumerate(key_list):
                if attack_type == "T1":
                    response["KEY" + str(index)] = response.pop(key)
                if attack_type == "T2":
                    if key == "id" or key == "ID" or "ID" in key.upper():
                        response[key] = "id_" + str(response[key])
                if attack_type == "T3":
                    camel_case_key = to_camel_case(key)
                    response[camel_case_key] = response.pop(key)
                if attack_type == "T4":
                    snake_case_str = to_snake_case(key)
                    response[snake_case_str] = response.pop(key)

        observation_dict["response"] = response
        observation = json.dumps(observation_dict)
        print("attack_observation",observation)

    except json.decoder.JSONDecodeError as e:
        print("error:",e)
        print("wrong observation:",observation)
    return observation




class single_chain(base_search_method):
    """Implement of CoT method
    """
    def __init__(self,llm,io_func,extra_prefix="",process_id=0,attack = None,start_message_list=None):
        """extra_prefix and start_message_list is used in Reflection Algo"""
        super(single_chain, self).__init__(llm,io_func, process_id, callbacks=None)
        self.io_func = io_func
        self.llm = llm
        self.extra_prefix = extra_prefix
        self.start_message_list = start_message_list
        self.process_id = process_id
        self.attack = attack

        self.restart()
    def restart(self):
        self.status = 0
        self.try_list = []
        self.terminal_node = []

        self.query_count = 0 # number of interactions with openai
        self.total_tokens = 0
        self.success_count = 0

    def to_json(self, answer=False,process=True):
        if process:
            json_obj = {
                "win": self.status == 1,
                "try_count": len(self.try_list),
                "trys": self.try_list,
                "compare_candidates": [],
                "forward_args":self.forward_args,
            }
            for node in self.terminal_node:
                if node.pruned == False: # has final answer
                    json_obj["compare_candidates"].append(node.get_chain_result_from_this_node(use_messages=False))
        else:
            json_obj = {}

        if answer:
            json_obj["answer_generation"] = {
                "valid_data": False,
                "final_answer": "",
                "function": self.io_func.functions,
                "query_count": self.query_count,
                "total_tokens": self.total_tokens,
                "train_messages": [],
                "chain": [],
            }
            for node in self.terminal_node:
                if node.pruned == False:
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["final_answer"] = node.description
                    json_obj["answer_generation"]["train_messages"] = node.get_train_messages_from_this_node()
                    break
        return json_obj

    def to_json_single(self):
        """parse the last try
        Though the nodes are formed as a tree, We still know they are actually a chain
        """
        json_obj = {}
        tree_obj = self.terminal_node[-1].get_chain_result_from_this_node()
        json_obj["chain"] = tree_obj
        json_obj["win"] = self.status == 1
        return json_obj

    def start(self,single_chain_max_step,pass_at=1,answer=1):
        self.forward_args = locals()
        if "self" in self.forward_args.keys():
            self.forward_args.pop("self")

        for i in range(pass_at):
            if self.process_id == 0:
                print(f"[single_chain]try for the {i+1} time")
            self.tree = my_tree()
            self.tree.root.node_type = "Action Input"
            self.tree.root.io_state = deepcopy(self.io_func)
            out_node = self.do_chain(self.tree.root, single_chain_max_step)
            self.terminal_node.append(out_node)
            self.try_list.append(self.to_json_single())
            if out_node.io_state.check_success() == 1:
                self.status = 1
                self.success_count += 1
                if self.success_count >= answer:
                    return 1
        return 0


    def do_chain(self,now_node,single_chain_max_step):

        if self.start_message_list == None:
            system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION

            system = system.replace("{task_description}",self.io_func.task_description)
            self.tree.root.messages.append({"role":"system","content":system})

            user = FORMAT_INSTRUCTIONS_USER_FUNCTION
            user = user.replace("{input_description}",self.io_func.input_description)
            self.tree.root.messages.append({"role":"user","content":user})


        else:
            """In Reflection Algo, we startswith former trials and reflections, so the caller will give the start messages"""
            self.tree.root.messages = self.start_message_list
        
        now_node = self.tree.root
        while True:
            # recursively parse message into nodes
            self.llm.change_messages(now_node.messages)
            if isinstance(self.llm, LlamaModel):
                new_message,error_code,total_tokens = self.llm.parse(functions=self.io_func.functions,process_id=self.process_id)
            else:
                new_message, error_code, total_tokens = self.llm.parse(tools=self.io_func.functions,
                                                                   process_id=self.process_id)

            self.total_tokens += total_tokens
            self.query_count += 1
            assert new_message["role"] == "assistant"
            if "content" in new_message.keys() and new_message["content"] != None:
                temp_node = tree_node()
                temp_node.node_type = "Thought"
                temp_node.description = new_message["content"]
                child_io_state = deepcopy(now_node.io_state)
                
                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0 
                temp_node.messages = now_node.messages.copy()
                temp_node.father = now_node
                now_node.children.append(temp_node)
                temp_node.print(self.process_id)
                now_node = temp_node

                if error_code != 0:
                    now_node.observation_code = error_code
                    now_node.pruned = True


            if "tool_calls" in new_message.keys() and new_message["tool_calls"] != None and len(new_message["tool_calls"]) > 0:
                tool_calls = new_message["tool_calls"]
                if self.process_id == 0:
                    print("number of parallel calls:",len(tool_calls))
                for i in range(len(tool_calls)):
                    function_name = tool_calls[i]["function"]["name"]
                    temp_node = tree_node()
                    temp_node.node_type = "Action"
                    temp_node.description = function_name
                    child_io_state = deepcopy(now_node.io_state)

                    temp_node.io_state = child_io_state
                    temp_node.is_terminal = child_io_state.check_success() != 0
                    temp_node.messages = now_node.messages.copy()
                    temp_node.father = now_node
                    now_node.children.append(temp_node)

                    temp_node.print(self.process_id)
                    now_node = temp_node

                    function_input = tool_calls[i]["function"]["arguments"]
                    temp_node = tree_node()
                    temp_node.node_type = "Action Input"
                    temp_node.description = function_input
                    child_io_state = deepcopy(now_node.io_state)

                    observation, status = child_io_state.step(action_name=now_node.description,
                                                              action_input=function_input)
                    if self.attack and self.attack.startswith("T"):
                        temp_node.observation = attack_response(observation, self.attack)
                    else:
                        temp_node.observation = observation

                    temp_node.observation_code = status

                    temp_node.io_state = child_io_state
                    temp_node.is_terminal = child_io_state.check_success() != 0
                    temp_node.messages = now_node.messages.copy()
                    temp_node.father = now_node
                    now_node.children.append(temp_node)
                    temp_node.print(self.process_id)
                    now_node = temp_node

                    if status != 0:
                        if status == 4:
                            now_node.pruned = True
                        elif status == 1:  # hallucination api name
                            assert "tool_calls" in new_message.keys() and len(new_message["tool_calls"]) > 0
                            tool_calls[i]["function"]["name"] = "invalid_hallucination_function_name"

                    if i == 0:
                        now_node.messages.append(new_message)
                    if now_node.node_type == "Action Input":
                        now_node.messages.append({
                            "role": "tool",
                            # "name": new_message["function_call"]["name"],
                            "name": tool_calls[i]["function"]["name"],
                            "content": now_node.observation,
                            "tool_call_id": tool_calls[i]['id'],
                        })
            elif "function_call" in new_message.keys() and new_message["function_call"] != None and len(new_message["function_call"]) > 0:
                function_call = new_message["function_call"]
                function_name = function_call["name"]
                temp_node = tree_node()
                temp_node.node_type = "Action"
                temp_node.description = function_name
                child_io_state = deepcopy(now_node.io_state)

                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0
                temp_node.messages = now_node.messages.copy()
                temp_node.father = now_node
                now_node.children.append(temp_node)

                temp_node.print(self.process_id)
                now_node = temp_node

                # function_input = new_message["function_call"]["arguments"]
                function_input = function_call["arguments"]
                function_input = re.sub(r'\(.*?\)', '', function_input).rstrip()

                temp_node = tree_node()
                temp_node.node_type = "Action Input"
                temp_node.description = function_input
                child_io_state = deepcopy(now_node.io_state)

                observation, status = child_io_state.step(action_name=now_node.description,
                                                          action_input=function_input)
                if self.attack and self.attack.startswith("T"):
                    temp_node.observation = attack_response(observation, self.attack)
                else:
                    temp_node.observation = observation

                temp_node.observation_code = status

                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0
                temp_node.messages = now_node.messages.copy()
                temp_node.father = now_node
                now_node.children.append(temp_node)
                temp_node.print(self.process_id)
                now_node = temp_node

                if status != 0:
                    # return code refers to Downstream_tasks/rapidapi
                    if status == 4:
                        now_node.pruned = True
                    elif status == 1:  # hallucination api name
                        assert "function_call" in new_message.keys()
                        function_call["name"] = "invalid_hallucination_function_name"

                now_node.messages.append(new_message)
                if now_node.node_type == "Action Input":
                    now_node.messages.append({
                        "role": "user",
                        # "name": new_message["function_call"]["name"],
                        "name": function_call["name"],
                        "content": now_node.observation,
                        # "tool_call_id": tool_calls[i]['id'],
                    })

            else:
                now_node.messages.append(new_message)
            
            if now_node.get_depth() >= single_chain_max_step and not (now_node.is_terminal):
                now_node.pruned = True
            # import pdb; pdb.set_trace()
            
            if now_node.pruned or now_node.is_terminal:
                return now_node

    
