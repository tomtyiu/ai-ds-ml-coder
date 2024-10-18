import streamlit as st
import pandas as pd
from io import StringIO
from openai import OpenAI
import os
# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values 
# loading variables from .env file
load_dotenv() 

# Huggingface Token, make sure to go to https://huggingface.co/settings/tokens
HF_TOKEN = os.getenv('HF_TOKEN')


# Set up Hugging Face API client using OpenAI-like interface
client = OpenAI(
    base_url="https://e28zttakj3zbw604.us-east-1.aws.endpoints.huggingface.cloud/v1/", 
    api_key=HF_TOKEN
)

# Initialize session state for chatbot and data
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Function to communicate with the LLM using Hugging Face's OpenAI-like API
def chat_with_llm(query):
    chat_completion = client.chat.completions.create(
        model="tgi",  # Use your deployed model
        messages=[{"role": "user", "content": query}],
        top_p=0.6,
        temperature=0.6,
        max_tokens=300,
        stream=True,
        seed=None,
        frequency_penalty=None,
        presence_penalty=None
    )
    
    # Concatenate all the messages returned in the response
    response_content = ""
    for message in chat_completion:
        response_content +=message.choices[0].delta.content
    
    return response_content

# Function to load and process data
def load_data(uploaded_file):
    if uploaded_file is not None:
        st.session_state['data'] = pd.read_csv(uploaded_file)
        st.write(f"Loaded {st.session_state['data'].shape[0]} rows and {st.session_state['data'].shape[1]} columns.")
        return True
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
    query = ""  # Reset input field

# Display messages in chatbot format
for message in st.session_state['messages']:
    st.markdown(f"**You:** {message['user']}")
    st.markdown(f"**Assistant:** {message['assistant']}")

# EDA Button
st.write("### Generate EDA Report")
if st.button("Generate EDA Report"):
    if st.session_state['data'] is not None:
        response = chat_with_llm("Generate an EDA report for the dataset with html format.")
        st.markdown("#### EDA Report")
        st.markdown(response)
        st.session_state['messages'].append({"user": "Generate EDA Report", "assistant": response})
    else:
        st.error("Please upload a dataset first.")

# Train Model Button
st.write("### Train a Model")
model_name = st.text_input("Enter the model type (e.g., llama, mistral, gpt):", "mistralai/Mistral-7B-Instruct-v0.3")
target_column = st.text_input("Enter the target column for training:")

if st.button("Train Model"):
    if st.session_state['data'] is not None and target_column:
        query = f"Generate code to train a {model_name} model using the dataset with the target variable '{target_column}'. Provide the code."
        response = chat_with_llm(query)
        st.markdown("#### Model Training Response")
        st.markdown(response)
        st.session_state['messages'].append({"user": f"Train a {model_name} model", "assistant": response})
    else:
        st.error("Please upload a dataset and specify the target column first.")

# Hyperparameter Tuning
st.write("### Hyperparameter Tuning")
if st.button("Suggest Hyperparameter Tuning"):
    response = chat_with_llm("Suggest hyperparameter tuning for the random_forest model.")
    st.markdown("#### Hyperparameter Tuning Suggestions")
    st.markdown(response)
    st.session_state['messages'].append({"user": "Suggest Hyperparameter Tuning", "assistant": response})

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state['messages'] = []
    st.success("Conversation cleared.")
