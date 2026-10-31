import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# =====================================================================
# Page Config
# =====================================================================
st.set_page_config(page_title="C++ RAG Chatbot", layout="wide")
st.title("C++ RAG with Ollama LLM")

# =====================================================================
# Load & Split Documents
# =====================================================================
@st.cache_resource
def load_vectorstore():
    # Construct an absolute path relative to where this script is saved
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "cpp.txt")

    # Stop gracefully with a clear message if the text file is missing
    if not os.path.exists(file_path):
        st.error(f"File not found at: `{file_path}`. Please create or copy `cpp.txt` into that directory.")
        st.stop()

    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    final_docs = text_splitter.split_documents(documents)

    # Create embeddings and store them in a FAISS vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(final_docs, embeddings)

    return db

# =====================================================================
# Load Vectorstore & LLM
# =====================================================================
db = load_vectorstore()
llm = Ollama(model="llama3.1:8b")

# =====================================================================
# Chat Interface
# =====================================================================
user_question = st.text_input("Ask a question about C++:")

if user_question:
    with st.spinner("Thinking..."):
        # Retrieve relevant documents from vectorstore
        docs = db.similarity_search(user_question, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question based on the context below. If the question can't be answered based on the context, say "I don't know".

        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """

        # Generate a response using Ollama LLM
        response = llm.invoke(prompt)
        
        st.subheader("Answer:")
        st.write(response)