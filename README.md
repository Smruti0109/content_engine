# content_engine - PDF Comparison & Insights System
This repository provides a Content Engine built using LangChain, Streamlit, and Hugging Face tools. The system is designed to analyze and compare multiple PDF documents, specifically Form 10-K filings of multinational companies (Alphabet Inc., Tesla, Inc., and Uber Technologies, Inc.). It leverages Retrieval-Augmented Generation (RAG) techniques to retrieve, assess, and generate insights from the documents.

# Features
Document Parsing & Comparison:- Extracts and analyzes text from multiple PDF documents. Identifies and highlights differences between documents, such as financial figures, risk factors, and business descriptions.
Vector Store Ingestion:- Uses embeddings to convert document content into vectors for efficient querying. The vectors are stored in a FAISS vector store.
Query Engine:- A powerful query engine based on Retrieval-Augmented Generation (RAG), allowing users to ask questions about the documents. The system retrieves the most relevant document content to answer queries.
Local Language Model:- Utilizes a local instance of a Hugging Face language model for generating answers and insights based on the document content. No external APIs are required.
Chatbot Interface:- A conversational interface built using Streamlit, enabling users to interact with the system by asking questions related to the documents.

# Setup
Clone the repository:
git clone https://github.com/your-username/content-engine-pdf-comparison.git
cd content-engine-pdf-comparison
Install dependencies:
pip install -r requirements.txt
Run the application:
streamlit run app.py

# Technical Stack
Backend Framework: LangChain – A toolkit for building LLM applications with a focus on retrieval-augmented generation (RAG).
Frontend Framework: Streamlit – For building the web interface and user interaction.
Vector Store: FAISS – For efficiently storing and querying document embeddings.
Embedding Model: Sentence-Transformers from Hugging Face – A local embedding model used to generate vector representations of document content.
Local LLM: Hugging Face's DialoGPT – A local model for generating answers and insights from document content.

# Acknowledgements
LangChain for the powerful toolkit to build the retrieval system.
Streamlit for simplifying the creation of web interfaces.
Hugging Face for their transformer models and tools.
PyMuPDF for efficient PDF text extraction.
