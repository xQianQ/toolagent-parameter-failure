# toolagent-parameter-failure


## 1. Install Dependencies  
Ensure Python 3.9 or higher is installed. Run the following command to install required packages:  
```bash  
pip install -r requirements.txt  
```  

## 2. Configuration Files in `config` Folder  

### Example Configuration File Structure:  
```  
config/  
├── default.json       # Default configuration template  
```  

### Key Configuration Parameters (in JSON format):  
```json  
{
    "base_url": "API endpoint URL",
    "openai_key": "API key or model identifier",
    "backbone_model": "Backbone model type (e.g., llama, chatgpt_function)",
    "chatgpt_model": "Specific model name",
    "model_path": "Path to model files",
    "tool_root_dir": "Root directory for tool definitions",
    "lora": "Whether to use LoRA adaptation",
    "lora_path": "Path to LoRA model (if enabled)",
    "max_observation_length": "Maximum length of observations",
    "max_source_sequence_length": "Original maximum sequence length of the model",
    "max_sequence_length": "Extended maximum sequence length",
    "observ_compress_method": "Method to compress observations (truncate, filter, random)",
    "method": "Answer generation method (CoT@n, Reflexion@n, BFS, DFS, UCT_vote)",
    "input_query_file": "Path to input query file",
    "input_query_dir": "Directory containing input queries (optional)",
    "output_answer_file": "Path to output answers file",
    "toolbench_key": "ToolBench service API key",
    "rapidapi_key": "RapidAPI service API key",
    "use_rapidapi_key": "Whether to use custom RapidAPI service",
    "api_customization": "Whether to use customized API",
    "device": "Device to run the model on (cpu, cuda)",
    "attack": "Type of attack (if applicable)",
    "gt_data_file": "Path to ground truth data file",
    "service_url": "Service endpoint URL",
    "cuda_device": "CUDA device ID"
}
```  
- The `attack` parameter in the configuration file is used to define perturbations during the QA pipeline execution.   

#### **Default Value**  
- `attack: null`: No perturbations are applied. The pipeline runs normally with clean input and tool interactions.  


#### **Perturbation Methods**  

The `attack` parameter supports three categories of perturbations. Below is a detailed breakdown:  


| Attack Value | Perturbation Method |  
|--------------|------------|  
| `D1`         | RD         |  
| `D2`         | RE         |  
| `D3`         | WD         |  
| `D4`         | SD         |  
| `D5`         | CO         |  
| `D6`         | WT         |  
| `Q1`         | RPF        |  
| `Q2`         | RPL        |  
| `Q3`         | CP         |  
| `Q4`         | AN         |  
| `T1`         | FK         |  
| `T2`         | AP         |  
| `T3`         | CK         |  
| `T4`         | UK         |  
| `T5`         | CF         |  


Here's a step-by-step guide on how to run:


### **How to Use**

#### **1. Ensure the RapidAPI Service is Running**  
- **Purpose**: The RapidAPI service is required to invoke external tools/APIs during the QA pipeline execution.  
- **Steps**:  
  - Start the RapidAPI service server.  
  - Verify that the service is accessible at the configured endpoint (e.g., `http://localhost:8080/rapidapi`).  
  - Ensure all required tools/APIs are registered and configured in the service.  
  - For more details, please refer to the [ToolBench](https://github.com/OpenBMB/ToolBench).

#### **2. Set the `PYTHONPATH` Environment Variable**  
- **Purpose**: The `PYTHONPATH` ensures Python can locate modules and packages in the project root directory.  
- **Command (Project Root Directory)**:  
  ```bash  
  export PYTHONPATH=./  
  ```  


#### **3. Run the Pipeline Script**  
- **Command**:  
  ```bash  
  python toolbench/inference/qa_pipeline.py  
  ```  
- **Notes**:  
  - **Configuration**: The script reads configuration from `config/default.json` by default. 
   
  - **Output**: Results are saved to the directory specified in `output_answer_file` (e.g., `./answers`).  

  
