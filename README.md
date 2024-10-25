# AI Data Science Assistant using Streamlit

## Overview
This application is an AI-powered Data Science Assistant built using Streamlit. It integrates OpenAI's language models to provide interactive support for data analysis, model training, exploratory data analysis (EDA), and data visualization. Users can upload datasets and interact with a chatbot that generates insights and responses for data processing and visualization tasks. Additionally, the assistant can generate comprehensive EDA reports and visualizations based on the uploaded data.

## Features
- **Upload and Load Dataset**: Upload CSV files, which are processed and made available for analysis.
- **OpenAI Model Integration**: Select from a range of OpenAI models to interact with the chatbot for data science-related queries.
- **Chatbot Assistant**: Ask questions or give commands to the assistant related to dataset analysis, EDA, and more.
- **EDA Reports**: Automatically generate a detailed exploratory data analysis report for the uploaded dataset.
- **Data Visualization**: Create visualizations (Bar and Line graphs) based on selected dataset columns.
- **Session Management**: Retains conversation history and dataset information throughout the session.
- **Clear Conversation**: Clear chatbot conversation history to start fresh.
- **Preview Dataset**: View a preview of the uploaded dataset.

## Requirements
- **Python 3.7+**
- Required Libraries:
  - `streamlit`
  - `pandas`
  - `matplotlib`
  - `openai`
  - `python-dotenv`

## Installation
1. Clone the repository:

```bash
   git clone <repo_url>
   cd <repo_directory>
```
Setup python virtual environment for safety
```bash
pip install virtualenv
virtualenv <directory>
cd <directory>\Scripts\activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```


Create a .env file in the root directory and add your OpenAI API key:

```bash
makefile
OPENAI_API_KEY=your_openai_api_key_here
```
Run the Streamlit app:

```bash
streamlit run app.py
```


## Usage Instructions
1. **OpenAI API Key** Input OpenAI API Key: Enter your OpenAI API key in the sidebar to enable interaction with the assistant.
2. **Upload Dataset**: Use the file uploader to upload a CSV file. The dataset will be loaded, and basic information will be displayed.
3. **Chat with the Assistant**: Ask questions or enter commands in the chatbox to interact with the AI assistant. It can generate dataset summaries, model suggestions, or other relevant insights.
4. **Generate EDA Report**: Click "Generate Comprehensive EDA Report" to get an automatic exploratory data analysis report of the dataset.
5. **Data Visualization**: Select a graph type (Bar or Line) and a column from the dataset to generate the respective visualization.
6. **Clear Conversation**: Use the "Clear Conversation" button to reset the chat history.
7. **Preview Dataset**: View the first few rows of the uploaded dataset in the preview section.

## Key Functionalities
Chat Function with LLM
The app allows users to communicate with a large language model (LLM) using OpenAI's API. Queries can be related to data analysis, visualization, or general commands, and the assistant will respond with relevant outputs or actions.

## Dataset Upload and Processing
CSV files can be uploaded, and the data is processed using pandas. The app provides a preview and summary of the dataset upon loading.

## EDA Report Generation
An automatic EDA report summarizes key statistical metrics for the dataset, including both categorical and numerical data, generated on-demand by the AI assistant.

## Data Visualization
The app supports Bar and Line chart generation for selected columns using matplotlib. Users can easily switch between visualization types and columns.

## Error Handling
The application provides user-friendly error messages for:

Incorrect or missing OpenAI API keys.
Issues during dataset loading or processing.
Invalid model selections.
Customization
To customize the app:

## Modify the list of available AI models in the model_options variable.
Add additional graph types or analysis tools to enhance the visualization capabilities.
Expand the chatbot's functionality to address more specific data science tasks.

## Monaco Editor
The st_monaco component from the streamlit_monaco package enables integration of the Monaco Editor (the editor used in Visual Studio Code) into Streamlit applications. This package allows for a flexible code editing experience within Streamlit, with support for syntax highlighting across various programming languages, dynamic resizing, and theme selection (dark or light mode based on Streamlit's theme settings). The editor can be customized for language-specific editing, and the content can be accessed or manipulated with Streamlit's built-in functions.
 
## Acknowledgements
OpenAI for the powerful AI models used in this application.
Streamlit for providing the framework for building interactive web applications.

## License
This project is licensed under the MIT License.
