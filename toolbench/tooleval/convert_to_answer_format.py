"""
Data preprocessing
"""
import argparse
import json
import os
from evaluation import ExecutionGraph, ExecutionNode
import random

class DataProcessor:
    def __init__(self, method, answer_dir, output_path):
        self.method = method
        self.answer_dir = answer_dir
        self.output_path = output_path
        random.seed(42)

    def generate_init_message_node(self, eg, functions, query):
        init_node = ExecutionNode(
            role='system',
            message="You are an AI assistant that can use tools to handle user queries."
        )
        eg.set_init_node(init_node)

        node = ExecutionNode(role='user', message=query)
        eg.add_node(node)
        eg[init_node, node] = None
        return node

    def process_valid_data(self, answer_generation):
        conversation = answer_generation['train_messages'][-1]
        functions = answer_generation['function']
        query = answer_generation['query']
        eg = ExecutionGraph()
        last_node = self.generate_init_message_node(eg, functions, query)

        index = 2
        while index < len(conversation):
            message = conversation[index]
            role = message['role']
            if role in {'system', 'user', 'function', 'tool'}:
                index += 1
                continue
            elif role == 'assistant':
                if 'function_call' in message and message['function_call']:
                    node = ExecutionNode(role='tool', message={
                        'name': message['function_call']['name'],
                        'arguments': message['function_call']['arguments'],
                        'response': conversation[index+1]['content'] if message['function_call']['name'] != 'Finish' else ''
                    })
                    index += 1
                elif 'tool_calls' in message and message['tool_calls']:
                    for tc in message['tool_calls']:
                        name = tc['function']['name']
                        arguments = tc['function']['arguments']
                        if name == 'Finish':
                            node = ExecutionNode(role='tool', message={
                                'name': name,
                                'arguments': arguments,
                                'response': ''
                            })
                            break
                        else:
                            for msg in conversation[index+1:]:
                                if msg['role'] == 'tool' and msg.get('tool_call_id') == tc['id']:
                                    node = ExecutionNode(role='tool', message={
                                        'name': name,
                                        'arguments': arguments,
                                        'response': msg['content']
                                    })
                                    eg.add_node(node)
                                    eg[last_node, node] = None
                                    last_node = node
                                    break
                else:
                    node = ExecutionNode(
                        role='assistant',
                        message=message['content']
                    )
            else:
                raise NotImplementedError(f'Unknown role: {role}')

            index += 1
            if last_node != node:
                eg.add_node(node)
                eg[last_node, node] = None
                last_node = node

        eg = eg.reduce_graph_to_sequence()

        return {
            'query': query,
            'available_tools': functions,
            'answer': {
                'method': self.method,
                'total_steps': eg.node_count,
                'final_answer': answer_generation['final_answer'],
                'answer_details': eg.convert_to_dict()
            }
        }

    def process_invalid_data(self, data_dict):
        answer_generation = data_dict['answer_generation']
        functions = answer_generation['function']
        query = answer_generation['query']
        eg = ExecutionGraph()
        last_node = self.generate_init_message_node(eg, functions, query)

        if 'CoT' in self.method:
            trail = random.choice(data_dict["trys"])
            for message in trail['chain']:
                if message['node_type'] == 'Action':
                    node = ExecutionNode(role='tool', message={
                        'name': message['description'],
                        'arguments': trail['chain'][message['next']]['description'],
                        'response': trail['chain'][message['next']]['observation']
                    })
                elif message['node_type'] == 'Thought':
                    node = ExecutionNode(
                        role='assistant',
                        message=message['description']
                    )
                else:
                    raise NotImplementedError(f"Unknown node_type: {message['node_type']}")

                eg.add_node(node)
                eg[last_node, node] = None
                last_node = node
            eg = eg.reduce_graph_to_sequence()
        elif 'DFS' in self.method:
            def dfs(root):
                if not root['children']:
                    node = ExecutionNode(role=root['node_type'], message=root)
                    eg.add_node(node)
                    return node
                else:
                    child_nodes = [dfs(node) for node in root['children']]
                    root_node = ExecutionNode(role=root['node_type'], message=root)
                    eg.add_node(root_node)
                    for child_node in child_nodes:
                        eg.add_edge(root_node, child_node)
                    return root_node

            for node in data_dict['tree']['tree']['children']:
                eg[last_node, dfs(node)] = None

            def purify_graph(node):
                if node.role == 'Action':
                    for adj_node_id in eg.get_adjacent_node(node):
                        adj_node = eg[adj_node_id]
                        if adj_node.role == 'Action Input':
                            node.role = 'tool'
                            node.message = {
                                'name': node.message['description'],
                                'arguments': adj_node.message['description'],
                                'response': adj_node.message['observation']
                            }
                            to_nodes = eg.edges.pop(adj_node.node_id, {})
                            eg.edges[node.node_id].update(to_nodes)
                            break
                elif node.role == 'Thought':
                    node.role = 'assistant'
                    node.message = node.message['description']
                elif node.role in {'system', 'user'}:
                    pass
                else:
                    raise Exception(f'Unknown role: {node.role}')

                for adj_node_id in eg.get_adjacent_node(node):
                    purify_graph(eg[adj_node_id])

            purify_graph(last_node)
            eg = eg.reduce_graph_to_sequence()
        else:
            raise NotImplementedError(f'Unknown method: {self.method}')

        return {
            'query': query,
            'available_tools': functions,
            'answer': {
                'method': self.method,
                'total_steps': eg.node_count,
                'final_answer': answer_generation['final_answer'],
                'answer_details': eg.convert_to_dict()
            }
        }

    def process(self):
        answer_dict = {}
        for filename in os.listdir(self.answer_dir):
            if filename.endswith('.json') and self.method in filename:
                qid = filename.split('_')[0]
                file_path = os.path.join(self.answer_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data_dict = json.load(f)
                    if not data_dict['answer_generation']['valid_data']:
                        answer_dict[qid] = self.process_invalid_data(data_dict)
                    else:
                        answer_dict[qid] = self.process_valid_data(data_dict['answer_generation'])
                except Exception as e:
                    print(f"Skipping file {filename} due to error: {e}")

        print(f'Converted {len(answer_dict)} answers')
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(answer_dict, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_dir', type=str, default="/newdisk/public/XQ/toolagent-parameter-failure/result/virtual_chatgpt_cot/G1_instruction", help='Directory containing answer files')
    parser.add_argument('--method', type=str, default="CoT@1", help='Method name')
    parser.add_argument('--output', type=str, default="/newdisk/public/XQ/toolagent-parameter-failure/result/virtual_chatgpt_cot/G1_instruction/converted_answers.json", help='Output file path')
    args = parser.parse_args()

    processor = DataProcessor(args.method, args.answer_dir, args.output)
    processor.process()