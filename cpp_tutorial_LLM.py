import streamlit as st

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- Streamlit UI ----------------

st.title("LangChain RAG Demo with llama3.1:8b")
st.write("Ask questions about your C++ knowledge source.")

input_text = st.text_input("What question do you have in mind?")

# ---------------- Load Knowledge Source ----------------

@st.cache_resource
def load_vector_db():
    loader = TextLoader("cpp.txt", encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db


# ---------------- LLM ----------------

llm = OllamaLLM(
    model="llama3.1:8b",
    temperature=0.2
)

output_parser = StrOutputParser()

# ---------------- Prompt Template ----------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful C++ programming assistant.
Answer the user's question using the retrieved context.
Explain concepts clearly and provide examples when useful.
If the context does not contain the answer, say so clearly.

Retrieved context:
{context}"""
    ),
    ("human", "{question}")
])

# ---------------- RAG Chain ----------------

try:
    vector_db = load_vector_db()
    retriever = vector_db.as_retriever(
        search_kwargs={"k": 3}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | output_parser
    )

    # ---------------- Generate Answer ----------------

    if input_text:
        with st.spinner("Searching documents and generating answer..."):
            response = chain.invoke(input_text)

        st.subheader("Answer")
        st.write(response)

except Exception as e:
    st.error(f"Error: {e}")
    st.info(
        "Check that cpp.txt exists and Ollama is running "
        "with llama3.1:8b installed."
    )