import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
from openai import OpenAI 
import os
from dotenv import load_dotenv 
# Created by Thomas Yiu 
# AI Data Science Assistant using Streamlit

# Load environment variables
load_dotenv()

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

#OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set up OpenAI API client
client = OpenAI(api_key=openai_api_key)

# Initialize session state for chatbot and dataset
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Function to communicate with LLM using OpenAI API
def chat_with_llm(query, model, dataset_summary=None):
    if dataset_summary:
        query = f"{dataset_summary}\n\n{query}"  # Include dataset context in the query
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}]
        )
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
st.title("LLM-Powered Data Science Assistant")
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
    # Generate a summary of the dataset
    if st.session_state['data'] is not None:
        dataset_summary = st.session_state['data'].describe(include='all').to_string()  # Summarize the dataset
        response = chat_with_llm(query, selected_model, dataset_summary)
    else:
        response = chat_with_llm(query, selected_model)

    st.session_state['messages'].append({"user": query, "assistant": response})
    st.text_input("Ask something (or type a command):", value="", key="input")  # Reset input field

# Display messages in a chatbot-like format
for message in st.session_state['messages']:
    st.markdown(f"**You:** {message['user']}")
    st.markdown(f"**Assistant:** {message['assistant']}")

# EDA Report Button
st.write("### Generate Comprehensive EDA Report")
if st.button("Generate Comprehensive EDA Report"):
    if st.session_state['data'] is not None:
        # Generate a summary of the dataset for the assistant
        dataset_summary = st.session_state['data'].describe(include='all').to_string()  # Summarize the dataset
        response = chat_with_llm("Please provide a comprehensive EDA report for the uploaded dataset.", selected_model, dataset_summary)
        
        st.markdown("#### Comprehensive EDA Report")
        st.markdown(response, unsafe_allow_html=True)
        st.session_state['messages'].append({"user": "Generate Comprehensive EDA Report", "assistant": response})
    else:
        st.error("Please upload a dataset first.")

# Data Visualization Section
st.write("### Data Visualization")
graph_type = st.selectbox("Select Graph Type", ["Bar", "Line"])
column = st.selectbox("Select Column for Visualization", st.session_state['data'].columns.tolist() if st.session_state['data'] is not None else [])

if st.button("Generate Graph"):
    if st.session_state['data'] is not None and column:
        plt.figure(figsize=(10, 5))
        
        if graph_type == "Bar":
            st.session_state['data'][column].value_counts().plot(kind='bar')
            plt.title(f'Bar Graph of {column}')
        elif graph_type == "Line":
            st.session_state['data'][column].plot(kind='line')
            plt.title(f'Line Graph of {column}')

        plt.xlabel(column)
        plt.ylabel('Frequency' if graph_type == "Bar" else 'Value')
        plt.grid()
        st.pyplot(plt)  # Display the plot in Streamlit
        st.session_state['messages'].append({"user": f"Generate {graph_type} graph for {column}", "assistant": "Graph generated."})
    else:
        st.error("Please upload a dataset and select a column first.")

# Clear Conversation Button
if st.button("Clear Conversation"):
    st.session_state['messages'] = []
    st.success("Conversation cleared.")

# Display Dataset Button
if st.session_state['data'] is not None:
    st.write("### Dataset Preview")
    st.dataframe(st.session_state['data'].head())
