import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


SOURCE_FILE = Path(__file__).with_name("cpp.txt")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="C++ RAG chatbot", page_icon=":material/smart_toy:")
st.title("C++ RAG chatbot")
st.caption("Ask questions about C++ Introduction using the local reference document.")


@st.cache_resource
def load_vectorstore():
    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(f"Reference document not found: {SOURCE_FILE}")

    loader = TextLoader(str(SOURCE_FILE), encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
    )
    return FAISS.from_documents(chunks, embeddings)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

try:
    vectorstore = load_vectorstore()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(
        "The reference document could not be loaded. "
        f"Check that `{SOURCE_FILE.name}` exists and is readable.",
        icon=":material/error:",
    )
    st.exception(error)
    st.stop()

if prompt := st.chat_input("Ask a C++ question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    documents = vectorstore.similarity_search(prompt, k=3)
    response = "Here are the most relevant sections from the C++ reference:"

    with st.chat_message("assistant"):
        st.markdown(response)
        for index, document in enumerate(documents, start=1):
            with st.expander(f"Relevant section {index}"):
                st.markdown(document.page_content)

    st.session_state.messages.append({"role": "assistant", "content": response})