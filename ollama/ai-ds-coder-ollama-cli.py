import os
import json
import pandas as pd
import argparse
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.chat_models.huggingface import ChatHuggingFace
from transformers import BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage
from pathlib import Path
from langchain_ollama import ChatOllama
llm = ChatOllama(
    model="hf.co/legolasyiu/Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.003-128K-code-ds-Q8_0-GGUF:latest",
    temperature=0.6,
    max_new_tokens=128000,
    # other params...
)

#chat_model = ChatHuggingFace(llm=llm)
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="LLM-based Data Science CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Interactive mode
    subparsers.add_parser("interactive", help="Start an interactive LLM session")

    # Load data command
    load_parser = subparsers.add_parser("load", help="Load a dataset")
    load_parser.add_argument("--file", type=str, required=True, help="Path to the data file")

    # Suggest task (preprocessing, hyperparameter tuning, feature engineering)
    suggest_parser = subparsers.add_parser("suggest", help="LLM suggestions for preprocessing, hyperparameter tuning, or feature engineering")
    suggest_parser.add_argument("--task", type=str, required=True, choices=["preprocessing", "hyperparameter_tuning", "feature_engineering"],
                                help="Task type for LLM suggestion")

    # Train model command
    train_parser = subparsers.add_parser("train", help="Train a machine learning model")
    train_parser.add_argument("--model", type=str, required=True, help="Model name (e.g., random_forest, xgboost)")
    train_parser.add_argument("--file", type=str, required=True, help="Path to the data file")
    train_parser.add_argument("--target", type=str, required=True, help="Target variable for model training")

    # Generate EDA report
    eda_parser = subparsers.add_parser("eda", help="Generate an EDA report")
    eda_parser.add_argument("--file", type=str, required=True, help="Path to the data file")
    eda_parser.add_argument("--plot", type=str, help="Specify plot types (e.g., 'all', 'scatter', 'heatmap')")

    # Evaluate trained model
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model", type=str, required=True, help="Trained model to evaluate")
    eval_parser.add_argument("--file", type=str, required=True, help="Path to the test data file")
    eval_parser.add_argument("--target", type=str, required=True, help="Target variable for evaluation")

    # Cross-validation
    cv_parser = subparsers.add_parser("crossval", help="Perform cross-validation")
    cv_parser.add_argument("--model", type=str, required=True, help="Model name (e.g., random_forest, xgboost)")
    cv_parser.add_argument("--file", type=str, required=True, help="Path to the data file")
    cv_parser.add_argument("--target", type=str, required=True, help="Target variable for cross-validation")

    return parser.parse_args()

# Interactive mode
def interactive_mode():
    print("=======================================================================")
    print("Starting Interactive Session with LLM")
    print("=======================================================================")
    query = ""
    while query.lower() != "exit":
        query = input("USER>> ")
        if query.lower() == "exit":
            break
        chat_with_llm(query)

# Load and process data
def load_data(file_path):
    print(f"Loading data from {file_path}...")
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.csv':
        data = pd.read_csv(file_path)
        print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    else:
        print(f"Unsupported file type: {ext}")
        return None

# Pass task to LLM for suggestions (preprocessing, hyperparameter tuning, feature engineering)
def suggest_task(task):
    if task == "preprocessing":
        query = "Suggest the best preprocessing steps for a CSV dataset."
    elif task == "hyperparameter_tuning":
        query = "Suggest hyperparameter tuning steps for a random forest model."
    elif task == "feature_engineering":
        query = "Suggest feature engineering techniques for the dataset."
    
    chat_with_llm(query)

# Pass model training to the LLM
def train_model_with_llm(model_name, file_path, target):
    try:
        data = load_data(file_path)
        if data is None:
            raise ValueError("Data loading failed")
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in the dataset")

        # Prepare the task for LLM
        query = f"Generate python code to train a {model_name} model using the dataset with the target variable '{target}'. Provide the code."
        response = chat_with_llm(query)

        # Execute the code generated by LLM (ensure PythonREPL runs it)
        get_python_repl(response)
    except Exception as e:
        print(f"Error during model training: {str(e)}")

# Generate EDA report using the LLM
def generate_eda_with_llm(file_path, plot_type):
    data = load_data(file_path)
    if data is not None:
        # Ask the LLM to generate an EDA report
        query = f"Generate an EDA report using python code for the dataset. Include {plot_type} plots if applicable."
        response = chat_with_llm(query)

        # Execute the EDA code generated by LLM
        get_python_repl(response)
    else:
        print("Failed to load data for EDA")

# Evaluate trained model with LLM
def evaluate_model_with_llm(model_name, file_path, target):
    try:
        data = load_data(file_path)
        if data is None:
            raise ValueError("Data loading failed")
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in the dataset")

        # Prepare the task for LLM
        query = f"Generate python code to evaluate the {model_name} model using the dataset with the target variable '{target}'. Provide evaluation metrics."
        response = chat_with_llm(query)

        # Execute the code generated by LLM for evaluation
        get_python_repl(response)
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")

# Perform cross-validation using LLM
def cross_validation_with_llm(model_name, file_path, target):
    try:
        data = load_data(file_path)
        if data is None:
            raise ValueError("Data loading failed")
        if target not in data.columns:
            raise ValueError(f"Created python code to target column '{target}' not found in the dataset")

        # Prepare the task for LLM
        query = f"Created python code to perform cross-validation for the {model_name} model using the dataset with the target variable '{target}'. Provide the code."
        response = chat_with_llm(query)

        # Execute the cross-validation code generated by LLM
        get_python_repl(response)
    except Exception as e:
        print(f"Error during cross-validation: {str(e)}")

# Function to communicate with the LLM
def chat_with_llm(query):
    messages = [
         SystemMessage(content="""
        Environment: ipython. Tools: brave_search, wolfram_alpha. Cutting Knowledge Date: December 2023. Today Date: 24 August 2024\n
        You are a coding assistant with expert with everything\n
        Ensure any code you provide can be executed \n
        with all required imports and variables defined. List the imports.  Structure your answer with a description of the code solution. \n
        write only the code. do not print anything else.\n
        use ipython for search tool. \n
        debug code if error occurs. \n
        Here is the user question: {query}
        """
        ),
        HumanMessage(content=query),
    ]
    response = llm.invoke(messages)
    print(f"LLM Response:\n{response.content}")
    return response.content

# Execute Python code using a Python REPL
def get_python_repl(ai_msg):
    python_repl = PythonREPL()
    result = python_repl.run(ai_msg)
    print(result)

# Main function
def main():
    args = parse_args()

    if args.command == "interactive":
        interactive_mode()
    elif args.command == "load":
        load_data(args.file)
    elif args.command == "suggest":
        suggest_task(args.task)
    elif args.command == "train":
        train_model_with_llm(args.model, args.file, args.target)
    elif args.command == "eda":
        generate_eda_with_llm(args.file, args.plot)
    elif args.command == "evaluate":
        evaluate_model_with_llm(args.model, args.file, args.target)
    elif args.command == "crossval":
        cross_validation_with_llm(args.model, args.file, args.target)
    else:
        print("Invalid command. Use --help for available commands.")

if __name__ == "__main__":
    main()
