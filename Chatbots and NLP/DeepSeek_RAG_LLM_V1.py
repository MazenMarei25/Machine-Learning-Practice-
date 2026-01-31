import os
import json
import streamlit as st
import ollama
import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from sentence_transformers import SentenceTransformer, util

# Directories for storage
DB_DIR = "./chroma_db"
PDF_DIR = "./uploaded_pdfs"
CHAT_HISTORY_FILE = "./uploaded_pdfs/chat_history.json"

# Ensure necessary directories exist
os.makedirs(PDF_DIR, exist_ok=True)

# Initialize Streamlit UI
st.set_page_config(page_title="Local LLM Chat", layout="wide")
st.title("Local LLM with RAG")

# Sidebar file uploader
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # For restriction checking

# Load or create a ChromaDB instance
chroma_client = chromadb.PersistentClient(path=DB_DIR)
vector_db = Chroma(client=chroma_client, embedding_function=embeddings)

# Load chat history from file
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    return {"messages": [], "restriction": None, "response_length": 150, "response_tone": "Neutral", "temperature": 0.7}

# Save chat history to file
def save_chat_history():
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump({
            "messages": st.session_state["messages"],
            "restriction": st.session_state["restriction"],
            "response_length": st.session_state["response_length"],
            "response_tone": st.session_state["response_tone"],
            "temperature": st.session_state["temperature"]
        }, file)

# Initialize session state
chat_data = load_chat_history()
st.session_state.setdefault("messages", chat_data["messages"])
st.session_state.setdefault("restriction", chat_data["restriction"])
st.session_state.setdefault("response_length", chat_data["response_length"])
st.session_state.setdefault("response_tone", chat_data["response_tone"])
st.session_state.setdefault("temperature", chat_data["temperature"])

# Process PDFs and store in database
def process_pdfs(files):
    documents = []
    for file in files:
        file_path = os.path.join(PDF_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)
        documents.extend(split_docs)
    vector_db.add_documents(documents)
    st.sidebar.success("Documents processed and stored in database!")
    save_chat_history()

# If new PDFs are uploaded, process them
if uploaded_files:
    process_pdfs(uploaded_files)

# List stored PDF documents
pdf_files = os.listdir(PDF_DIR)
st.sidebar.subheader(f"Stored PDFs ({len(pdf_files)})")
for pdf in pdf_files:
    if pdf != "chat_history.json":
        st.sidebar.write(f"- {pdf}")

# Load Ollama model with GPU offloading
llm = ollama.chat(model="deepseek-r1:1.5b")
retriever = vector_db.as_retriever()
qa_chain = ConversationalRetrievalChain(llm=llm, retriever=retriever)

# Sidebar restriction input
st.sidebar.subheader("Set Topic Restriction")
restriction_input = st.sidebar.text_input("Enter restriction (e.g., 'Only topics related to Human Resources')")
if st.sidebar.button("Apply Restriction"):
    st.session_state["restriction"] = restriction_input
    st.sidebar.success(f"Restriction applied: {restriction_input}")
    save_chat_history()

if st.sidebar.button("Remove Restriction"):
    st.session_state["restriction"] = None
    st.sidebar.success("Topic restriction removed.")
    save_chat_history()

# Sidebar response customization
st.sidebar.subheader("Response Settings")
st.session_state["response_length"] = st.sidebar.slider("Max words in response", 50, 500, st.session_state["response_length"])
st.session_state["response_tone"] = st.sidebar.selectbox("Response Tone", ["Neutral", "Formal", "Casual", "Verbose + Formal", "Concise"], index=["Neutral", "Formal", "Casual", "Verbose + Formal", "Concise"].index(st.session_state["response_tone"]))
st.session_state["temperature"] = st.sidebar.slider("Temperature (0.1 = more deterministic, 1.0 = more random)", 0.1, 1.0, 0.7, 0.1)

# Display chat history
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

# Chat input
user_input = st.chat_input("Ask something...")
if user_input:
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})
    chat_history = [(msg["role"], msg["content"]) for msg in st.session_state["messages"]]
    response = qa_chain.run(user_input, chat_history=chat_history)
    st.chat_message("assistant").write(response)
    st.session_state["messages"].append({"role": "assistant", "content": response})
    save_chat_history()
