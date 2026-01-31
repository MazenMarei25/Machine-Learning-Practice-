import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.memory.vectorstore_token_buffer_memory import (ConversationVectorStoreTokenBufferMemory)

# Load the model
mdl = "llama3.2:1b"
LANGUAGE_MODEL = OllamaLLM(model=mdl)

embedder = OllamaEmbeddings(model="mxbai-embed-large:latest")

chroma = Chroma(collection_name="memorytest",
                embedding_function=embedder,
                )

retriever = chroma.as_retriever(
        search_type="similarity_score_threshold",
)

conversation_memory = ConversationVectorStoreTokenBufferMemory(
        return_messages=True,
        llm=LANGUAGE_MODEL,
        retriever=retriever,
        max_token_limit = 2000,
        split_chunk_size= 1000,
)



if chroma.count() == 0:
    conversation_memory.save_context({"human": "user", "humancontent": "initializing"},
                                    {"ai": "assistant", "aicontent": "this is prechat, this message is for initialization"}
    )

# Define the prompt template (RAG context can be added if needed)
PROMPT_TEMPLATE = """
You are an expert research assistant. Be factual and provide your references.Recall info from chat history
Query: {user_query}
History: {history}
Answer:
"""
st.title("Simple Chat")

# Initialize UI chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from  UI history on app rerun
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
    loaded_memory = conversation_memory.load_memory_variables({"input": prompt})["messages"]
    # Create conversation prompt
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        response_stream = response_chain.stream({"user_query": prompt})
        full_response = st.write_stream(response_stream)  # Stream response
        
    # Ensure response is stored properly
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    conversation_memory.save_context({"human": "user", "humancontent": prompt},
                                    {"ai": "assistant", "aicontent": full_response})
