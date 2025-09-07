import streamlit as st
import sys
import os

# Add parent folder to sys.path so pipeline can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import load_vectorstore, build_rag_chain

# Load pipeline
st.title("🔎 RAG Chatbot")
st.write("Ask me anything about the website data you upload...")

@st.cache_resource
def get_chain():
    vectorstore = load_vectorstore()
    return build_rag_chain(vectorstore)

rag_chain = get_chain()

# User input
query = st.text_input("💭 Your Question:")
if query:
    with st.spinner("Thinking..."):
        answer = rag_chain.invoke(query)
    st.markdown(f"**Answer:** {answer}")
