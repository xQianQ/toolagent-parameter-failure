import json
import argparse

def extract_and_format_json(json_file_path,category_name):
    # Load the JSON data from the file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    # print(data)
    # Extract the required information
    tool_description = data.get('tool_description', '')
    name = data.get('name', '')
    api_list = data.get('api_list', [])
    api_list = api_list[:6]

    # Create the desired output format
    output = {
        "tool_description": tool_description,
        "name": name,
        "api_list": []
    }

    # Process each API in the api_list
    for api in api_list:
        api_name = api.get('name', '')
        api_url = api.get('url', '')
        api_description = api.get('description', '')
        api_method = api.get('method', '')
        api_required_parameters = api.get('required_parameters', [])
        api_optional_parameters = api.get('optional_parameters', [])

        # Create the API dictionary
        api_dict = {
            "name": api_name,
            "url": api_url,
            "description": api_description,
            "method": api_method,
            "required_parameters": api_required_parameters,
            "optional_parameters": api_optional_parameters,
            "tool_name": name,
            "category_name": category_name
        }

        # Add the API dictionary to the output
        output["api_list"].append(api_dict)

    return output

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_files',required =False,default= ['/newdisk/public/XQ/experience/LLM_Tool_Learning/data/data/toolenv/tools/Data/dog_breeds.json'], type=list, help='Path to the JSON files')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Extract and format the JSON data
    for json_file in args.json_files:
        cat_name = json_file.split('/')[-2]
        output = extract_and_format_json(json_file,cat_name)
        # Save the output to a new JSON file
        output_file_path = f'{json_file}.apilist.json'


if __name__ == '__main__':
    main()
