from pathlib import Path

import streamlit as st

from rag import Answer, KnowledgeBase, load_documents


ROOT = Path(__file__).parent
SAMPLE_DIR = ROOT / "sample_documents"

st.set_page_config(page_title="College knowledge assistant", page_icon=":material/school:", layout="wide")
st.title("College knowledge assistant")
st.caption("Ask questions about the official documents. Every answer is grounded in retrieved passages.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []


@st.cache_resource
def build_sample_kb() -> KnowledgeBase:
    return KnowledgeBase(load_documents(sorted(SAMPLE_DIR.glob("*"))))


def render_answer(answer: Answer) -> None:
    if not answer.found:
        st.warning(answer.text, icon=":material/search_off:")
    elif answer.conflict:
        st.warning(answer.text, icon=":material/warning:")
    else:
        st.markdown(answer.text)
    if answer.sources:
        st.markdown("**Supporting sources**")
        for index, result in enumerate(answer.sources, start=1):
            location = f", page {result.chunk.page}" if result.chunk.page else ""
            with st.expander(f"{index}. {result.chunk.source}{location} · relevance {result.score:.2f}"):
                st.write(result.chunk.text)


with st.sidebar:
    st.header("Knowledge base")
    uploaded = st.file_uploader(
        "Add PDF or text documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload at least five official documents for a realistic demo.",
    )
    if uploaded:
        st.session_state.uploaded_docs = uploaded
    if st.session_state.uploaded_docs:
        st.success(f"{len(st.session_state.uploaded_docs)} uploaded document(s) ready.")
        if st.button("Use sample documents", icon=":material/refresh:"):
            st.session_state.uploaded_docs = []
            st.rerun()
    else:
        st.info("Using the five included sample documents.")
    st.divider()
    st.caption("Retrieval uses local TF-IDF search. No student question is sent to an external service.")

try:
    if st.session_state.uploaded_docs:
        temp_dir = ROOT / ".uploaded_documents"
        temp_dir.mkdir(exist_ok=True)
        chunks = []
        for file in st.session_state.uploaded_docs:
            target = temp_dir / file.name
            target.write_bytes(file.getvalue())
            chunks.extend(load_documents([target]))
        kb = KnowledgeBase(chunks)
    else:
        kb = build_sample_kb()
except (OSError, RuntimeError, ValueError) as error:
    st.error(f"Could not build the knowledge base: {error}", icon=":material/error:")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_answer(message["answer"])

if prompt := st.chat_input("Ask a question about the college documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    answer = kb.answer(prompt)
    with st.chat_message("assistant", avatar=":material/school:"):
        render_answer(answer)
    st.session_state.messages.append({"role": "assistant", "answer": answer})
