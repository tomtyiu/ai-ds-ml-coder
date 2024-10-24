import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
from streamlit_monaco import st_monaco

# Load environment variables (for OpenAI API key)
load_dotenv()

# Initialize OpenAI API key from environment variables
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Create two columns for layout
col1, col2 = st.columns(2)

# Upload CSV in left column (col1)
with col1:
    st.write("### Upload Dataset for Data Analysis")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # Load the data into a pandas DataFrame and store in session state
    if uploaded_file:
        st.session_state['data'] = pd.read_csv(uploaded_file)
        st.write(f"Loaded {st.session_state['data'].shape[0]} rows and {st.session_state['data'].shape[1]} columns.")
    else:
        st.session_state['data'] = None
        st.warning("Please upload a CSV file to continue.")

# Left column: Monaco editor to input code for plotting
with col1:
    st.title("Code-to-Graph with Streamlit and Monaco")

    # Function to execute code with the provided dataset
    def execute_code(code, data):
        try:
            exec(code, {"data": data, "pd": pd, "go": go, "plt": plt})  # Pass the 'data' DataFrame into Monaco code
            return True, None
        except Exception as e:
            return False, str(e)

    # Monaco editor with sample code referring to the 'data' DataFrame
    code = st_monaco(
        value="""import plotly.graph_objects as go
if data is not None:
    fig = go.Figure(data=[go.Bar(x=data['Column1'], y=data['Column2'])])
    fig.show()
else:
    print("No data available to plot.")""",
        language="python",
        height="300px",
    )

    # Button to execute the Monaco code
    if st.button("Execute and Graph"):
        if st.session_state['data'] is not None:
            success, error = execute_code(code, st.session_state['data'])
            if success:
                st.success("Code executed successfully!")
            else:
                st.error(f"Error: {error}")
        else:
            st.error("Please upload a dataset first.")

# Right column: Chatbot and model interaction
with col2:
    st.title("LLM-Powered Data Science Assistant")

    # OpenAI API key input
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    
    # Only initialize the OpenAI client if API key is provided
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)
    else:
        client = None
        st.error("Please provide a valid OpenAI API key.")

    # Initialize session state for chatbot and messages
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Function to interact with OpenAI LLM
    def chat_with_llm(query, model, dataset_summary=None):
        if dataset_summary:
            query = f"{dataset_summary}\n\n{query}"
        if client:
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": query}]
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                st.error(f"Error communicating with the model: {e}")
                return "An error occurred while trying to get a response."
        else:
            return "Please provide a valid OpenAI API key."

    # Dropdown for selecting OpenAI model
    model_options = [
        "o1-preview-2024-09-12", 
        "o1-mini-2024-09-12",  
        "chatgpt-4o-latest",  
        "gpt-4o-mini-2024-07-18"
    ]
    selected_model = st.selectbox("Select OpenAI Model", model_options)

    # Chatbot interface
    st.write("### Chat with the Assistant")
    query = st.text_input("Ask something (or type a command):")

    # Display chatbot conversation
    if query:
        dataset_summary = st.session_state['data'].describe(include='all').to_string() if st.session_state['data'] is not None else None
        response = chat_with_llm(query, selected_model, dataset_summary)
        st.session_state['messages'].append({"user": query, "assistant": response})
        st.text_input("Ask something (or type a command):", value="", key="input")  # Reset input field

    # Display chatbot message history
    for message in st.session_state['messages']:
        st.markdown(f"**You:** {message['user']}")
        st.markdown(f"**Assistant:** {message['assistant']}")

    # EDA Report button
    st.write("### Generate Comprehensive EDA Report")
    if st.button("Generate Comprehensive EDA Report"):
        if st.session_state['data'] is not None:
            dataset_summary = st.session_state['data'].describe(include='all').to_string()
            response = chat_with_llm("Please provide a comprehensive EDA report for the uploaded dataset.", selected_model, dataset_summary)
            st.markdown("#### Comprehensive EDA Report")
            st.markdown(response, unsafe_allow_html=True)
            st.session_state['messages'].append({"user": "Generate Comprehensive EDA Report", "assistant": response})
        else:
            st.error("Please upload a dataset first.")

    # Data Visualization
    st.write("### Data Visualization")
    
    if st.session_state['data'] is not None:
        # Select columns for visualization
        selected_columns = st.multiselect("Select Column(s) for Visualization", st.session_state['data'].columns.tolist())

        # Row filtering options
        st.write("#### Filter Rows (Optional)")
        row_filter_type = st.selectbox("Filter rows by:", ["None", "Index Range", "Conditional Filter"])
        filtered_data = st.session_state['data']

        if row_filter_type == "Index Range":
            start_index = st.number_input("Start Index", min_value=0, max_value=len(filtered_data) - 1, value=0)
            end_index = st.number_input("End Index", min_value=0, max_value=len(filtered_data) - 1, value=len(filtered_data) - 1)
            filtered_data = filtered_data.iloc[start_index:end_index + 1]

        elif row_filter_type == "Conditional Filter":
            condition_column = st.selectbox("Select Column for Condition", filtered_data.columns.tolist())
            condition_value = st.text_input("Enter Value for Filtering")
            if condition_value:
                filtered_data = filtered_data[filtered_data[condition_column].astype(str) == condition_value]

        # Select graph type
        graph_type = st.selectbox("Select Graph Type", ["Bar", "Line", "Scatter", "Histogram", "Pie"])

        # Generate graph
        if st.button("Generate Graph"):
            if len(selected_columns) > 0:
                plt.figure(figsize=(10, 5))

                if graph_type == "Bar":
                    for col in selected_columns:
                        filtered_data[col].value_counts().plot(kind='bar', alpha=0.5)
                    plt.title(f'Bar Graph of {", ".join(selected_columns)}')

                elif graph_type == "Line":
                    for col in selected_columns:
                        filtered_data[col].plot(kind='line', alpha=0.5)
                    plt.title(f'Line Graph of {", ".join(selected_columns)}')

                elif graph_type == "Scatter" and len(selected_columns) >= 2:
                    filtered_data.plot(kind='scatter', x=selected_columns[0], y=selected_columns[1])
                    plt.title(f'Scatter Plot of {selected_columns[0]} vs {selected_columns[1]}')

                elif graph_type == "Histogram":
                    filtered_data[selected_columns].plot(kind='hist', bins=30, alpha=0.5)
                    plt.title(f'Histogram of {", ".join(selected_columns)}')

                elif graph_type == "Pie" and len(selected_columns) == 1:
                    filtered_data[selected_columns[0]].value_counts().plot(kind='pie', autopct='%1.1f%%')
                    plt.title(f'Pie Chart of {selected_columns[0]}')

                plt.xlabel('Index')
                plt.ylabel('Frequency' if graph_type in ["Bar", "Histogram"] else 'Value')
                plt.grid()
                st.pyplot(plt)

                st.session_state['messages'].append({
                    "user": f"Generate {graph_type} graph for {', '.join(selected_columns)} with row filter: {row_filter_type}",
                    "assistant": f"{graph_type} graph generated for {', '.join(selected_columns)}"
                })
            else:
                st.error("Please select column(s) for visualization.")
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state['messages'] = []
        st.success("Conversation cleared.")
