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

# Layout for other functionality
st.write("### Generate EDA Report")
if st.button("Generate EDA Report"):
    if st.session_state['data'] is not None:
        response = chat_with_llm("Write a EDA report for the dataset in HTML format.", selected_model)
        st.markdown("#### EDA Report")
        st.markdown(response, unsafe_allow_html=True)
        st.session_state['messages'].append({"user": "Generate EDA Report", "assistant": response})
    else:
        st.error("Please upload a dataset first.")

# Additional functionalities can go below...
