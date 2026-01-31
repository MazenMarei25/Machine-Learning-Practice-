import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.memory.vectorstore_token_buffer_memory import ConversationVectorStoreTokenBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
import os
import torch
torch.classes.__path__ = []
#os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
#What is the Performance Tier of Olivia Bennett ?
mdl = "llama3.2:1b"
LANGUAGE_MODEL = OllamaLLM(model=mdl)
embedder = OllamaEmbeddings(model="mxbai-embed-large:latest")

chroma_history = Chroma(
    collection_name="history",
    embedding_function=embedder,
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory="./history_db_2",
)

chroma_RAG = Chroma(
    collection_name="RAG",
    embedding_function=embedder,
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory="./RAG_db_2",
)


retriever_history = chroma_history.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'k': 2, 'score_threshold': 0.5},
)

retriever_RAG = chroma_RAG.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'k': 2, 'score_threshold': 0.5},
)


def load_documents_from_directory(directory_path):
    file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(('.xls', '.csv','.pdf'))]
    loader = DoclingLoader(file_path=file_paths, export_type=ExportType.MARKDOWN)
    return loader.load()

def chunk_documents(documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(documents)

def index_documents(document_chunks):
    chroma_RAG.add_documents(document_chunks)

conversation_memory = ConversationVectorStoreTokenBufferMemory(
    return_messages=False,
    llm=LANGUAGE_MODEL,
    retriever=retriever_history,
    max_token_limit=1000,
    split_chunk_size=1000,
)

UPLOAD_DIR = "D:/Coding Projects/langflow/pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a file")
    if uploaded_file is not None:
        saved_file_path = save_uploaded_file(uploaded_file)
        st.write(f"Uploaded file saved at: {saved_file_path}")

        # Load, chunk, and index the uploaded document
        documents = load_documents_from_directory(UPLOAD_DIR)
        document_chunks = chunk_documents(documents)
        index_documents(document_chunks)
        st.write(f"Indexed {len(document_chunks)} document chunks.")

def clear_chat_history():
    # Replace with your ChromaDB collection deletion logic
    chroma_history.reset_collection()
    st.session_state.messages = []  # Clears Streamlit session history
    st.rerun()  # Refreshes the app

# Dialog function for confirmation
@st.dialog("Confirm Deletion", width="small")

def confirm_deletion():
    st.write("Are you sure you want to delete the entire chat history? This action is irreversible.")
    if st.button("Yes, delete"):
        clear_chat_history()
    if st.button("Cancel"):
        st.stop()
        
with st.sidebar:
    if st.button("Clear Chat History"):
        confirm_deletion()

PROMPT_TEMPLATE = """
Your answers are short and concise.Answer all user questions. 
Use the important information from the memory context and the retrieved context to formulate your answer to the user query.
Query: {user_query}
Memory: {memory_context}
Retrieved Documents: {retrieved_docs}
Answer:
"""

st.title("RAG Chat With Memory")

# Initialize UI chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from UI history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to UI chat history 
    st.session_state.messages.append({"role": "user", "content": prompt})
    loaded_memory = conversation_memory.load_memory_variables({"input": prompt}) #['history']

    related_docs = retriever_RAG.get_relevant_documents(prompt)
    retrieved_docs = "\n\n".join([doc.page_content for doc in related_docs])

    # Create conversation prompt
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        response_stream = response_chain.stream({
    "user_query": prompt,
    "memory_context": loaded_memory,
    "retrieved_docs": retrieved_docs
        })
        full_response = st.write_stream(response_stream)  # Stream response
        
    # Ensure response is stored properly
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    conversation_memory.save_context({"Human": prompt},
                                    {"AI": full_response})
    conversation_memory.save_remainder()








