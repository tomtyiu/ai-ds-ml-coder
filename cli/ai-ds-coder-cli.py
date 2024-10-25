## Created by Thomas Yiu
import os
import json
import pandas as pd
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from transformers import BitsAndBytesConfig
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import json
from pathlib import Path
from pprint import pprint
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredExcelLoader
# importing os module for environment variables
import os
# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values 
# loading variables from .env file
load_dotenv() 

# Huggingface Token, make sure to go to https://huggingface.co/settings/tokens
HF_TOKEN = os.getenv('HF_TOKEN')

#quantization 4bit for quick response 
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

llm = HuggingFacePipeline.from_model_id(
    model_id="EpistemeAI/Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.003-128K-code-ds",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=2048,
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
    ),
    model_kwargs={"quantization_config": quantization_config},
)

chat_model = ChatHuggingFace(llm=llm)

from google.colab import userdata

os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')

# Command interface for file uploading and LLM interaction
def interactive_menu():
    print("=======================================================================")
    print("Welcome to the Interactive Command Interface")
    print("=======================================================================")
    print("Options:")
    print("1. Upload CSV/TXT/PDF files and chat LLM to analysis")
    print("2. Chat with the LLM")
    print("Enter 'exit' to leave")

    while True:
        choice = input("Please select an option (1, 2, 3): ").strip()

        if choice == "1":
            upload_file()
        elif choice == "2":
            query = ""
            print("Starting Chat with LLM. Type 'bye' to exit chat.")
            while query.lower() != "bye":
                query = input("USER>> ")
                if query.lower() == "bye":
                    break
                chatbot(query)  # LLM chat functionality defined in your code
        elif choice.lower() == "exit":
            print("Exiting the interface. Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")



def upload_file():
    file_path = input("Enter the path of the Excel/PDF file to upload: ").strip()

    if not os.path.exists(file_path):
        print("File does not exist. Please try again.")
        return

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.xlsx':
      loader = UnstructuredExcelLoader(file_path, mode="elements")
      docs = loader.load()
      print(docs[0].metadata)
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
      splits = text_splitter.split_documents(docs)
      vectorstore = InMemoryVectorStore.from_documents(
          documents=splits, embedding=OpenAIEmbeddings()
          )

      # Retrieve and generate using the relevant snippets of the blog.
      retriever = vectorstore.as_retriever()
      #prompt = hub.pull("rlm/rag-prompt")
      from langchain.chains import create_retrieval_chain
      from langchain.chains.combine_documents import create_stuff_documents_chain
      from langchain_core.prompts import ChatPromptTemplate

      system_prompt = (
          "You are an assistant for question-answering tasks. "
          "Use the following pieces of retrieved context to answer the question.\n"
          "If you don't know the answer, say that you don't know. \n"
          "Use three sentences maximum and keep the answer concise. \n"
          "You are also a coding assistant with retrieved context"
          "\n\n"
          "{context}"
      )

      prompt = ChatPromptTemplate.from_messages(
          [
              ("system", system_prompt),
              ("human", "{input}"),
          ]
      )


      question_answer_chain = create_stuff_documents_chain(llm, prompt)
      rag_chain = create_retrieval_chain(retriever, question_answer_chain)



      query = ""
      while query != "bye":
        query = input("USER>>")
        results = rag_chain.invoke({"input":  query})  #"What was Nike's revenue in 2023?"
        #results
        print(results["context"][0].page_content)
        print(results["context"][0].metadata)
    elif ext == '.txt':
        with open(file_path, 'r') as file:
            data = file.read()
        print("File uploaded successfully. Here's the content of the text file:")
        print(data[:500])  # Showing first 500 characters for preview
    elif ext == '.pdf':
        #data = json.loads(Path(file_path).read_text())
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        print(docs[0].page_content[0:100])
        print(docs[0].metadata)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = InMemoryVectorStore.from_documents(
            documents=splits, embedding=OpenAIEmbeddings()
            )

        # Retrieve and generate using the relevant snippets of the blog.
        retriever = vectorstore.as_retriever()
        #prompt = hub.pull("rlm/rag-prompt")
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )


        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)



        query = ""
        while query != "bye":
          query = input("USER>>")
          results = rag_chain.invoke({"input":  query})  #"What was Nike's revenue in 2023?"
          #results
          print(results["context"][0].page_content)
          print(results["context"][0].metadata)
          input_user=input("if it is code, execute the code:yes or no")
          if input_user == "yes":
            get_python_repl(results["context"][0].page_content)
          query = input("USER>>")
          results = rag_chain.invoke({"input":  query})  #"What was Nike's revenue in 2023?"

# python repl to execute code, make run docker or virtual environment
def get_python_repl(ai_msg):
    python_repl = PythonREPL()
    # You can create the tool to pass to an agent
    print(python_repl.run(ai_msg))

# Assuming your chatbot code is wrapped in this function
def chatbot(input_user):
    from langchain_core.messages import (
        HumanMessage,
        SystemMessage,
    )

    messages = [
        SystemMessage(content=
        """
        Environment: ipython. Tools: brave_search, wolfram_alpha. Cutting Knowledge Date: December 2023. Today Date: 24 August 2024\n
        You are a coding assistant with expert with everything\n
        Ensure any code you provide can be executed \n
        with all required imports and variables defined. List the imports.  Structure your answer with a description of the code solution. \n
        write only the code. do not print anything else.\n
        use ipython for search tool. \n
        debug code if error occurs. \n
        Here is the user question: {question}
        """
        ),
        HumanMessage(
            content=input_user
        ),
    ]

    ai_msg = chat_model.invoke(messages)
    print(ai_msg.content)
    get_python_repl(ai_msg.content)
    print ("double checking>>>")
    from langchain_core.messages import (
        HumanMessage,
        SystemMessage,
    )

    messages = [
        SystemMessage(content=
        """
        Environment: ipython. Tools: brave_search, wolfram_alpha. Cutting Knowledge Date: December 2023. Today Date: 24 Auguest 2024\n
        You are a debug assistant. find bugs and fix the bugs nad refactor\n
        Ensure any code you provide can be executed \n
        with all required imports and variables defined. List the imports.  Structure your answer with a description of the code solution. \n
        write only the code. do not print anything else.\n
        use ipython for search tool. \n
        debug code if error occurs. \n
        if no error, please provide no code \n
        if error, fix the code and provide the code\n
        """
        ),
        HumanMessage(
            content=ai_msg.content
        ),
    ]

    ai_msg_2 = chat_model.invoke(messages)
    print(ai_msg_2.content)
    get_python_repl(ai_msg_2.content)

    query = ""
    while query != "bye":
      query = input("USER>>")
      chatbot(query)
      #print(f"Chatbot>>:{response}\n")

# Start the interactive menu
interactive_menu()
