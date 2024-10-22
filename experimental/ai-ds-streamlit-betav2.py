import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face Token, ensure you have it from https://huggingface.co/settings/tokens
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set up Hugging Face API client using OpenAI-like interface
client = OpenAI()

# Initialize session state for chatbot and dataset
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Streamlit page configuration
st.set_page_config(page_title="LLM Data Science Assistant", page_icon="🤖", layout="centered")

# Adding some basic CSS to style the app
st.markdown("""
    <style>
    .main {
        background-color: #f0f0f5;
        color: #333333;
    }
    .stTextInput>div>input {
        background-color: white;
        border-radius: 5px;
        border: 1px solid #cccccc;
        color: #333333;
    }
    .stButton>button {
        background-color: #f9f9f9;
        color: #4CAF50;
        border: none;
        padding: 10px 24px;
        font-size: 14px;
        border-radius: 12px;
    }
    .stButton>button:hover {
        background-color: #dddddd;
        color: #333333;
    }
    .assistant-response {
        background-color: #f1f1f1;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: #333333;
    }
    .user-message {
        background-color: #e0e0e0;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# Function to communicate with LLM using Hugging Face API
def chat_with_llm(query, model):
    try:
        # Call the API without streaming
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}]
        )
        # Get the response content
        response_content = completion.choices[0].message.content
        return response_content.strip()
    except Exception as e:
        st.error(f"Error communicating with the model: {e}")
        return "An error occurred while trying to get a response."

# Function to load and process data from CSV
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            st.session_state['data'] = pd.read_csv(uploaded_file)
            st.write(f"Loaded {st.session_state['data'].shape[0]} rows and {st.session_state['data'].shape[1]} columns.")
            return True
        except Exception as e:
            st.error(f"Error loading the file: {e}")
            return False
    return False

# Streamlit Layout
st.title("LLM-Powered Data Science Assistant 🤖")
st.write("Upload your dataset and interact with the assistant to train models, generate EDA reports, and more.")

# Dropdown for selecting OpenAI model
model_options = [
    "o1-preview-2024-09-12", 
    "o1-mini-2024-09-12",  
    "chatgpt-4o-latest",  
    "gpt-4o-mini-2024-07-18"
]
selected_model = st.selectbox("Select OpenAI Model", model_options)

# Chatbot container
st.write("### Chat with the Assistant")
query = st.text_input("Ask something (or type a command):")

# Upload CSV file
st.write("### Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Button to process uploaded file
if uploaded_file:
    if load_data(uploaded_file):
        st.success("File uploaded and loaded successfully.")

# Display chatbot interaction
if query:
    response = chat_with_llm(query, selected_model)  # Use the selected model here
    st.session_state['messages'].append({"user": query, "assistant": response})
    st.text_input("Ask something (or type a command):", value="", key="input")  # Reset input field

# Display messages in a chatbot-like format with custom CSS
for message in st.session_state['messages']:
    st.markdown(f'<div class="user-message">**You:** {message["user"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="assistant-response">**Assistant:** {message["assistant"]}</div>', unsafe_allow_html=True)


# EDA Report Button
st.write("### Generate EDA Report")
if st.button("Generate EDA Report"):
    if st.session_state['data'] is not None:
        response = chat_with_llm("Write a EDA report for the dataset in HTML format.", selected_model)
        st.markdown("#### EDA Report")
        st.markdown(response, unsafe_allow_html=True)
        st.session_state['messages'].append({"user": "Generate EDA Report", "assistant": response})
    else:
        st.error("Please upload a dataset first.")

# Data Imputation Button
st.write("### Data Imputation")
if st.button("Suggest Data Imputation"):
    if st.session_state['data'] is not None:
        response = chat_with_llm("Suggest data imputation steps for missing values in the dataset.", selected_model)
        st.markdown("#### Data Imputation Suggestions")
        st.markdown(response)
        st.session_state['messages'].append({"user": "Suggest Data Imputation", "assistant": response})
    else:
        st.error("Please upload a dataset first.")

# Feature Engineering Button
st.write("### Feature Engineering")
if st.button("Suggest Feature Engineering"):
    if st.session_state['data'] is not None:
        response = chat_with_llm("Suggest feature engineering steps for the dataset.", selected_model)
        st.markdown("#### Feature Engineering Suggestions")
        st.markdown(response)
        st.session_state['messages'].append({"user": "Suggest Feature Engineering", "assistant": response})
    else:
        st.error("Please upload a dataset first.")

# Train Model Button
st.write("### Train a Model")
model_name = st.text_input("Enter the model type (e.g., Random Forest, XGBoost, Decision Tree):", "Random Forest")
target_column = st.text_input("Enter the target column for training:")

if st.button("Train Model"):
    if st.session_state['data'] is not None and target_column:
        query = f"Generate code to train a {model_name} model using the dataset with the target variable '{target_column}'. Provide the code."
        response = chat_with_llm(query, selected_model)
        st.markdown("#### Model Training Code")
        st.code(response, language="python")
        st.session_state['messages'].append({"user": f"Train a {model_name} model", "assistant": response})
    else:
        st.error("Please upload a dataset and specify the target column first.")

# Model Evaluation Button
st.write("### Model Evaluation")
if st.button("Evaluate Model"):
    if st.session_state['data'] is not None and target_column:
        query = f"Provide evaluation metrics for a {model_name} model trained on the target column '{target_column}'."
        response = chat_with_llm(query, selected_model)
        st.markdown("#### Model Evaluation Metrics")
        st.markdown(response)
        st.session_state['messages'].append({"user": "Evaluate Model", "assistant": response})
    else:
        st.error("Please upload a dataset and specify the target column first.")

# Cross-Validation Button
st.write("### Perform Cross-Validation")
if st.button("Perform Cross-Validation"):
    if st.session_state['data'] is not None and target_column:
        query = f"Generate code for performing cross-validation on a {model_name} model using the target column '{target_column}'."
        response = chat_with_llm(query, selected_model)
        st.markdown("#### Cross-Validation Code")
        st.code(response, language="python")
        st.session_state['messages'].append({"user": "Perform Cross-Validation", "assistant": response})
    else:
        st.error("Please upload a dataset and specify the target column first.")

# Clustering Button
st.write("### Apply Clustering")
if st.button("Apply Clustering"):
    if st.session_state['data'] is not None:
        response = chat_with_llm("Suggest clustering techniques (e.g., K-means) to apply to the dataset.", selected_model)
        st.markdown("#### Clustering Suggestions")
        st.markdown(response)
        st.session_state['messages'].append({"user": "Apply Clustering", "assistant": response})
    else:
        st.error("Please upload a dataset first.")

# Outlier Detection Button
st.write("### Outlier Detection")
if st.button("Detect Outliers"):
    if st.session_state['data'] is not None:
        response = chat_with_llm("Detect outliers in the dataset.", selected_model)
        st.markdown("#### Outlier Detection Suggestions")
        st.markdown(response)
        st.session_state['messages'].append({"user": "Detect Outliers", "assistant": response})
    else:
        st.error("Please upload a dataset first.")

# Data Visualization Button
st.write("### Data Visualization")
if st.button("Generate Data Visualizations"):
    if st.session_state['data'] is not None:
        response = chat_with_llm("Suggest basic data visualizations (e.g., histograms, scatter plots, correlation heatmaps) for the dataset.", selected_model)
        st.markdown("#### Data Visualization Suggestions")
        st.markdown(response)
        st.session_state['messages'].append({"user": "Generate Data Visualizations", "assistant": response})
    else:
        st.error("Please upload a dataset first.")

# Automated Report Button
st.write("### Generate Automated Report")
if st.button("Generate Automated Report"):
    if st.session_state['data'] is not None:
        response = chat_with_llm("Generate an automated report summarizing the dataset, models trained, and results.", selected_model)
        st.markdown("#### Automated Report")
        st.markdown(response, unsafe_allow_html=True)
        st.session_state['messages'].append({"user": "Generate Automated Report", "assistant": response})
    else:
        st.error("Please upload a dataset first.")

# Time Series Analysis Button
st.write("### Time Series Analysis")
if st.button("Perform Time Series Analysis"):
    if st.session_state['data'] is not None:
        response = chat_with_llm("Suggest time series analysis techniques for the dataset.", selected_model)
        st.markdown("#### Time Series Analysis Suggestions")
        st.markdown(response)
        st.session_state['messages'].append({"user": "Perform Time Series Analysis", "assistant": response})
    else:
        st.error("Please upload a dataset first.")

# Clear Conversation Button
if st.button("Clear Conversation"):
    st.session_state['messages'] = []
    st.success("Conversation cleared.")

# Display Dataset Button
if st.session_state['data'] is not None:
    st.write("### Dataset Preview")
    st.dataframe(st.session_state['data'].head())
