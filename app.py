import streamlit as st
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF for faster PDF processing
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pickle  # For saving and loading vector store
from hashlib import md5
import time  # For performance tracking
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Define CSS for styling
css = """
    <style>
        .chat-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            max-width: 700px;
            margin: auto;
            padding: 20px;
            background-color: #f8f8f8;
            border-radius: 8px;
        }
        .chat-box {
            display: flex;
            flex-direction: column;
            width: 100%;
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 8px;
            max-height: 500px;
            overflow-y: auto;
            margin-bottom: 10px;
        }
        .user-message {
            background-color: #D6E4FF;
            color: black;
            padding: 10px;
            border-radius: 8px;
            margin: 5px 0;
            align-self: flex-end;
            max-width: 80%;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        .bot-message {
            background-color: #E3E3E3;
            color: black;
            padding: 10px;
            border-radius: 8px;
            margin: 5px 0;
            align-self: flex-start;
            max-width: 80%;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        .input-container {
            display: flex;
            width: 100%;
            margin-top: 10px;
        }
        .input-box {
            width: 90%;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ccc;
        }
        .send-button {
            width: 10%;
            padding: 10px;
            background-color: #4CAF50;
            border-radius: 8px;
            color: white;
            border: none;
            cursor: pointer;
            font-weight: bold;
        }
        .send-button:hover {
            background-color: #45a049;
        }
    </style>
"""

# Load HuggingFace model and initialize embeddings
def load_huggingface_model(model_name="microsoft/DialoGPT-medium"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=512,
            max_new_tokens=150
        )
        return HuggingFacePipeline(pipeline=hf_pipeline)
    except Exception as e:
        st.error(f"Failed to load Hugging Face model: {str(e)}")
        return None

# Initialize embeddings
try:
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
except Exception as e:
    st.error(f"Error initializing embeddings: {str(e)}")

def get_pdf_text(pdf_docs):
    """Extract text from PDFs in parallel using PyMuPDF."""
    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(extract_text_from_pdf, pdf_docs))

    combined_text = "".join(results)
    st.info(f"PDF text extraction completed in {time.time() - start_time:.2f} seconds.")
    return combined_text

def extract_text_from_pdf(pdf_file):
    doc_text = ""
    try:
        with fitz.open("pdf", pdf_file.read()) as doc:
            for page_num in range(doc.page_count):
                page_text = doc[page_num].get_text("text")
                if page_text:
                    doc_text += page_text
                else:
                    st.warning(f"Warning: Page {page_num + 1} of {pdf_file.name} is empty or unreadable.")
    except Exception as e:
        st.error(f"Error reading {pdf_file.name}: {str(e)}")
    return doc_text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text(text)

def vectorstore_cache_path(text_chunks):
    hash_id = md5("".join(text_chunks).encode()).hexdigest()
    return f"vectorstore_{hash_id}.pkl"

def get_vectorstore(text_chunks):
    cache_path = vectorstore_cache_path(text_chunks)
    if os.path.exists(cache_path):
        st.info("Loading vectorstore from cache.")
        return load_vectorstore(cache_path)

    st.info("Generating new vectorstore...")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    save_vectorstore(vectorstore, cache_path)
    return vectorstore

def save_vectorstore(vectorstore, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(vectorstore, f)

def load_vectorstore(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def get_conversation_chain(vectorstore, hf_pipeline):
    memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=hf_pipeline,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        output_key="answer",  # Set to store only the answer in memory
        return_source_documents=True  # Allows source documents in the response
    )

def handle_userinput():
    """Process user input with the conversation chain."""
    if "user_question" not in st.session_state or not st.session_state.user_question:
        return

    user_question = st.session_state.user_question
    conversation_chain = st.session_state.conversation

    response = conversation_chain({"question": user_question})
    answer = response.get("answer", "I'm sorry, but I couldn't find an answer.")
    source_docs = response.get("source_documents", [])

    # Format response with source-based differentiation
    if source_docs:
        formatted_answer = f"{answer}\n\nSources:\n"
        for doc in source_docs:
            doc_title = doc.metadata.get("title", "Document")
            formatted_answer += f"- **{doc_title}**: {doc.page_content[:500]}...\n"
    else:
        formatted_answer = answer

    st.session_state.chat_history.append({"user": user_question, "bot": formatted_answer})
    st.session_state.user_question = ""  # Clear input

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    if "hf_pipeline" not in st.session_state:
        st.session_state.hf_pipeline = load_huggingface_model()
        if st.session_state.hf_pipeline is None:
            return

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_chat_history" not in st.session_state:
        st.session_state.show_chat_history = False

    st.header("Chat with multiple PDFs 📚")

    user_question = st.text_input("Ask a question about your documents:", key="user_question")
    st.button("Send", on_click=handle_userinput)

    if st.button("View Chat History"):
        st.session_state.show_chat_history = not st.session_state.show_chat_history

    with st.container():
        st.write('<div class="chat-container">', unsafe_allow_html=True)
        st.write('<div class="chat-box">', unsafe_allow_html=True)

        if st.session_state.show_chat_history and st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                st.write(f'<div class="user-message"><b>Question:</b> {chat["user"]}</div>', unsafe_allow_html=True)
                st.write(f'<div class="bot-message"><b>Answer:</b> {chat["bot"]}</div>', unsafe_allow_html=True)
        elif not st.session_state.show_chat_history and st.session_state.chat_history:
            chat = st.session_state.chat_history[-1]
            st.write(f'<div class="user-message"><b>Question:</b> {chat["user"]}</div>', unsafe_allow_html=True)
            st.write(f'<div class="bot-message"><b>Answer:</b> {chat["bot"]}</div>', unsafe_allow_html=True)

        st.write('</div>', unsafe_allow_html=True)
        st.write('</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", type=["pdf"], accept_multiple_files=True)
        if pdf_docs:
            try:
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore, st.session_state.hf_pipeline)
                st.success("Documents processed successfully! You can ask questions now.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return

if __name__ == '__main__':
    main()
