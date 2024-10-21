# Created by Thomas Yiu
# AI Data Science Assistant using Streamlit

import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face Token, ensure you have it from https://huggingface.co/settings/tokens
#HF_TOKEN = os.getenv('HF_TOKEN')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set up Hugging Face API client using OpenAI-like interface
client = OpenAI(
    base_url="https://api.openai.com/v1/chat/completions",  # Example base URL
    api_key=OPENAI_API_KEY 
)

# Initialize session state for chatbot and dataset
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Function to communicate with LLM using Hugging Face API
def chat_with_llm(query):
    chat_completion = client.chat.completions.create(
        model="o1-mini-2024-09-12",  # or OpenAI model or Episteme AI model, for super advance reasoning, use o1-preview
        messages=[{"role": "user", "content": query}],
        top_p=0.6,
        temperature=0.6,
        max_tokens=300,
        stream=True
    )

    # Collect response content
    response_content = ""
    for message in chat_completion:
        response_content += message.choices[0].delta.content

    return response_content

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
st.title("LLM-Powered Data Science Assistant")
st.write("Upload your dataset and interact with the assistant to train models, generate EDA reports, and more.")

# Chatbot container
st.write("### Chat with the assistant")
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
    response = chat_with_llm(query)
    st.session_state['messages'].append({"user": query, "assistant": response})
    st.text_input("Ask something (or type a command):", value="", key="input")  # Reset input field

# Display messages in a chatbot-like format
for message in st.session_state['messages']:
    st.markdown(f"**You:** {message['user']}")
    st.markdown(f"**Assistant:** {message['assistant']}")

# EDA Report Button
st.write("### Generate EDA Report")
if st.button("Generate EDA Report"):
    if st.session_state['data'] is not None:
        response = chat_with_llm("Generate an EDA report for the dataset in HTML format.")
        st.markdown("#### EDA Report")
        st.markdown(response, unsafe_allow_html=True)
        st.session_state['messages'].append({"user": "Generate EDA Report", "assistant": response})
    else:
        st.error("Please upload a dataset first.")

# Data Imputation Button
st.write("### Data Imputation")
if st.button("Suggest Data Imputation"):
    if st.session_state['data'] is not None:
        response = chat_with_llm("Suggest data imputation steps for missing values in the dataset.")
        st.markdown("#### Data Imputation Suggestions")
        st.markdown(response)
        st.session_state['messages'].append({"user": "Suggest Data Imputation", "assistant": response})
    else:
        st.error("Please upload a dataset first.")

# Feature Engineering Button
st.write("### Feature Engineering")
if st.button("Suggest Feature Engineering"):
    if st.session_state['data'] is not None:
        response = chat_with_llm("Suggest feature engineering steps for the dataset.")
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
        response = chat_with_llm(query)
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
        response = chat_with_llm(query)
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
        response = chat_with_llm(query)
        st.markdown("#### Cross-Validation Code")
        st.code(response, language="python")
        st.session_state['messages'].append({"user": "Perform Cross-Validation", "assistant": response})
    else:
        st.error("Please upload a dataset and specify the target column first.")

# Clustering Button
st.write("### Apply Clustering")
if st.button("Apply Clustering"):
    if st.session_state['data'] is not None:
        response = chat_with_llm("Suggest clustering techniques (e.g., K-means) to apply to the dataset.")
        st.markdown("#### Clustering Suggestions")
        st.markdown(response)
        st.session_state['messages'].append({"user": "Apply Clustering", "assistant": response})
    else:
        st.error("Please upload a dataset first.")

# Outlier Detection Button
st.write("### Outlier Detection")
if st.button("Detect Outliers"):
    if st.session_state['data'] is not None:
        response = chat_with_llm("Detect outliers in the dataset.")
        st.markdown("#### Outlier Detection Suggestions")
        st.markdown(response)
        st.session_state['messages'].append({"user": "Detect Outliers", "assistant": response})
    else:
        st.error("Please upload a dataset first.")

# Data Visualization Button
st.write("### Data Visualization")
if st.button("Generate Data Visualizations"):
    if st.session_state['data'] is not None:
        response = chat_with_llm("Suggest basic data visualizations (e.g., histograms, scatter plots, correlation heatmaps) for the dataset.")
        st.markdown("#### Data Visualization Suggestions")
        st.markdown(response)
        st.session_state['messages'].append({"user": "Generate Data Visualizations", "assistant": response})
    else:
        st.error("Please upload a dataset first.")

# Automated Report Button
st.write("### Generate Automated Report")
if st.button("Generate Automated Report"):
    if st.session_state['data'] is not None:
        response = chat_with_llm("Generate an automated report summarizing the dataset, models trained, and results.")
        st.markdown("#### Automated Report")
        st.markdown(response, unsafe_allow_html=True)
        st.session_state['messages'].append({"user": "Generate Automated Report", "assistant": response})
    else:
        st.error("Please upload a dataset first.")

# Time Series Analysis Button
st.write("### Time Series Analysis")
if st.button("Perform Time Series Analysis"):
    if st.session_state['data'] is not None:
        response = chat_with_llm("Suggest time series analysis techniques for the dataset.")
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
