import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Load the model
mdl = "granite3.2-vision:latest"
LANGUAGE_MODEL = OllamaLLM(model=mdl)

# Define the prompt template (RAG context can be added if needed)
PROMPT_TEMPLATE = """
You are an expert research assistant. Be factual and provide your references.

Query: {user_query}
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

    # Create conversation prompt
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        response_stream = response_chain.stream({"user_query": prompt})
        full_response = st.write_stream(response_stream)  # Stream response
        
    # Ensure response is stored properly
    st.session_state.messages.append({"role": "assistant", "content": full_response})
