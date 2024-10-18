# LLM-based Data Science CLI Tool

## Overview
This is a command-line tool powered by a Large Language Model (LLM) designed to assist with various data science tasks. It enables users to load datasets, perform exploratory data analysis (EDA), train machine learning models, and more. The LLM generates code to automate these workflows, allowing for easy integration into data science pipelines.

## Features
Interactive Mode: Chat with the LLM to receive real-time suggestions and code generation for data science tasks.
Data Loading: Load datasets (CSV format) for processing and analysis.
Task Suggestions: Get recommendations for data preprocessing, hyperparameter tuning, or feature engineering.
Model Training: Automatically generate and run code to train machine learning models.
EDA Reports: Generate EDA reports, including customizable plots.
Model Evaluation: Evaluate the performance of trained models.
Cross-Validation: Perform cross-validation for different models using LLM-generated code.
Prerequisites
Ensure the following dependencies are installed:

```bash
pip install langchain transformers huggingface_hub pandas argparse
```

Also, ensure you have set up the OPENAI_API_KEY environment variable:

```bash
export OPENAI_API_KEY=your_openai_api_key
```

## Installation
Clone the repository:

```bash
git clone https://github.com/yourusername/llm-data-science-cli.git
```

Navigate into the project directory:

bash
Copy code
cd llm-data-science-cli
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
This tool provides a command-line interface for performing various data science tasks. Below are the available commands:

1. Interactive Mode
Start an interactive session with the LLM:

bash
Copy code
python cli_tool.py interactive
This mode allows you to type in queries and get responses from the LLM.

2. Load Data
Load a dataset for further processing:

bash
Copy code
python cli_tool.py load --file path/to/your/dataset.csv
3. Task Suggestions
Ask the LLM to suggest steps for preprocessing, hyperparameter tuning, or feature engineering:

bash
Copy code
python cli_tool.py suggest --task preprocessing
Supported tasks:

preprocessing
hyperparameter_tuning
feature_engineering
4. Train a Model
Train a machine learning model using your dataset:

bash
Copy code
python cli_tool.py train --model random_forest --file path/to/your/dataset.csv --target target_column
Replace random_forest with your preferred model type (e.g., xgboost).

5. Generate EDA Report
Generate an EDA report with customizable plot types:

bash
Copy code
python cli_tool.py eda --file path/to/your/dataset.csv --plot all
Specify plot types such as scatter, heatmap, or all to include different visualizations.

6. Evaluate Model
Evaluate the performance of a trained model:

bash
Copy code
python cli_tool.py evaluate --model random_forest --file path/to/test_data.csv --target target_column
7. Cross-Validation
Perform cross-validation on a machine learning model:

bash
Copy code
python cli_tool.py crossval --model xgboost --file path/to/your/dataset.csv --target target_column
Command Help
To view help and available commands:

bash
Copy code
python cli_tool.py --help
How It Works
The tool uses an LLM from HuggingFace or other sources to generate Python code based on user input. The generated code is executed within a Python REPL environment to automate various data science tasks. Core functionalities include:

Chat Interaction: Send user queries to the LLM and receive Python code as a response.
Python REPL: Execute LLM-generated code directly in the terminal.
Data Processing: Handle loading and preprocessing of datasets (currently supports CSV).
Environment Variables
Make sure the OPENAI_API_KEY environment variable is set in your system. This key is required for the tool to communicate with the LLM.

bash
Copy code
export OPENAI_API_KEY=your_openai_api_key
Example Workflow
A typical workflow using this tool might look like:

Interactive Mode: Start a chat session with the LLM to explore the tool’s features.
Load Dataset: Load a dataset for analysis.
Task Suggestions: Get suggestions for preprocessing or feature engineering.
Model Training: Train a machine learning model with the LLM-generated code.
Generate EDA: Generate an EDA report to understand the dataset.
Model Evaluation: Evaluate the model’s performance.
Cross-Validation: Run cross-validation to assess model robustness.
Error Handling
The tool includes error handling for common issues such as unsupported file types or missing target columns. Error messages provide guidance for troubleshooting.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contributions
Contributions are welcome! Please feel free to submit issues, fork the repository, or create pull requests.

Contact
For any issues or questions, please contact the project maintainer or submit an issue in the GitHub repository.
