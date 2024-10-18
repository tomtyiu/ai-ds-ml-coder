# ai-ds-coder
Artificial Intelligence Data science coder for data mining, large data processing, EDA on large data, visualization of data

This AI DS Coder is fast data scientist coder assistant. It will produce the code and execute the code.  There is 2 modes.  RAG with files or direct code mode. 

# First we need to install
```shell
!pip install langchain_community langchain_core
!pip install transformers langchain_experimental langchain_huggingface langchain_core
!pip install bitsandbytes
!pip install --quiet --upgrade langchain langchain-community langchain-chroma langchain_openai langchain-text-splitters pypdf
!pip install --upgrade --quiet langchain-community unstructured openpyxl
```

# For example

### Input prompt example: 
#### User>>Create a code to load the excel file: /content/Open LLM benchmark comparison.xlsx, extract the data and make sure to extract by model vs the Average 	Ifeval	BBH	MATH Lvl 5	GPQA	MUSR	MMLU-PRO.  Plot the table

<img src="extract excel sheet and generate graph.JPG">
