## AI-ds-coder-cli
Artificial Intelligence Data science coder for data mining, large data processing, EDA on large data, visualization of data

This AI DS Coder is fast data scientist coder assistant. It will produce the code and execute the code.  There is 2 modes.  RAG with files or direct code mode. 

## Setup

Install dependencies
```shell
!pip install langchain_community langchain_core
!pip install transformers langchain_experimental langchain_huggingface langchain_core
!pip install bitsandbytes
!pip install --quiet --upgrade langchain langchain-community langchain-chroma langchain_openai langchain-text-splitters pypdf
!pip install --upgrade --quiet langchain-community unstructured openpyxl
```

Create Python Virtual Environment (For security. highly recommended)

```shell
#install nenv to your host Python
pip install virtualenv

#create a new project folder and run command
python<version> -m venv <virtual-environment-name>

mkdir projectA
cd projectA
python3.8 -m venv env

# activate Virtual Environment
source env/bin/activate

#Note that to activate your virtual environment on Widows, 
env/Scripts/activate.bat //In CMD
env/Scripts/Activate.ps1 //In Powershel

#check if it is working
pip list

#deactive  Virtual Environment
deactivate
```
## How to setup ollama
- download ollama -  [ollama](https://ollama.com/download)
- install ollama
- in command prompt>> ollama run hf.co/legolasyiu/Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.003-128K-code-ds-Q8_0-GGUF


## For AI data science coder: ai-ds-coder-ollama-cli.py
## commands cli menu:
```shell

                        Available commands
    interactive         Start an interactive LLM session
    load                Load a dataset
    suggest             LLM suggestions for preprocessing, hyperparameter tuning, or feature engineering
    train               Train a machine learning model
    eda                 Generate an EDA report
    evaluate            Evaluate a trained model
    crossval            Perform cross-validation

options:
  -h, --help            show this help message and exit
```



# For example

### Input prompt example: 
#### User>>Create a code to load the excel file: /content/Open LLM benchmark comparison.xlsx, extract the data and make sure to extract by model vs the Average 	Ifeval	BBH	MATH Lvl 5	GPQA	MUSR	MMLU-PRO.  Plot the table

<img src="extract excel sheet and generate graph.JPG">

### Nightly update

- 10/18 = updated ai coder local to ai coder-ollama-cli, provide instructions on how to setup ollama

#### When visit this site or clone this page, please press like this github
