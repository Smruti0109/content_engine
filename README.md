# Content Engine-PDF Comparison & Insights System

This repository provides a Content Engine for analyzing and comparing multiple PDF documents, specifically designed for Form 10-K filings of multinational companies like Alphabet Inc., Tesla, Inc., and Uber Technologies, Inc. Built using LangChain, Streamlit, and Hugging Face, this system leverages Retrieval-Augmented Generation (RAG) to retrieve, assess, and generate insights from documents.

## Features

- **Document Parsing & Comparison**  
   - Extracts and analyzes text from multiple PDF documents.
   - Identifies and highlights differences between documents, such as financial figures, risk factors, and business descriptions.

- **Vector Store Ingestion**  
   - Uses embeddings to convert document content into vectors for efficient querying.
   - Vectors are stored in a FAISS vector store for rapid retrieval.

- **Query Engine**  
   - A powerful query engine based on RAG, enabling users to ask detailed questions about the documents.
   - Retrieves the most relevant document content to generate informed responses.

- **Local Language Model**  
   - Utilizes a local instance of a Hugging Face language model to generate answers and insights directly from the document content.
   - Operates independently without requiring external API calls.

- **Chatbot Interface**  
   - An interactive conversational interface built with Streamlit, allowing users to ask questions about the documents in a chat-based format.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/content-engine-pdf-comparison.git
   cd content-engine-pdf-comparison

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
