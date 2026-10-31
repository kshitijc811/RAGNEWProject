# RAG and machine learning demos

This repository contains a set of small Python projects and notebooks for learning Streamlit, retrieval-augmented generation (RAG), and neural networks. The projects are independent: install the requirements f# RAG and machine learning demos

This repository is a collection of small Python applications, notebooks, and learning materials for retrieval-augmented generation (RAG), neural networks, and related topics. Each demo is independent; use its matching requirements file.

## Project structure

```text
RAGNEWProject/
├── app.py, rag.py                 # Main college-document RAG app and retrieval logic
├── CNNapp.py, cnn_model.py        # CNN image-classification app
├── chatBotDemo.py                # C++ document RAG demo
├── ragChatbotLLM.py               # C++ RAG demo with a local Ollama model
├── cpp_tutorial_LLM.py            # C++ RAG tutorial
├── streamlit_tutorial.py          # Streamlit examples
├── data/
│   ├── sample_documents/          # Sample source files for the college RAG app
│   └── reference/cpp.txt          # Source text for C++ RAG demos
├── docs/notes/                    # Edited course and project notes
├── notebooks/                     # Jupyter learning notebooks
├── papers/                        # Research papers and reading materials
├── tests/                         # Checks for the main RAG pipeline
└── requirements*.txt              # Separate dependency sets
```

## Applications

| Project | Entry point | Purpose |
|---|---|---|
| College document assistant | `app.py` | Streamlit app that searches sample or uploaded PDF, Markdown, and text documents. It presents supporting passages and declines low-relevance questions. |
| CNN image classifier | `CNNapp.py` | Trains a small CNN on CIFAR-10 or image folders in an uploaded ZIP, then predicts an uploaded image's class. |
| C++ RAG experiments | `chatBotDemo.py`, `ragChatbotLLM.py`, `cpp_tutorial_LLM.py` | Earlier LangChain examples using the C++ reference text; two use Ollama. |
| Streamlit tutorial | `streamlit_tutorial.py` | Examples of inputs, charts, forms, and file uploads. |
| RAG notebooks | `notebooks/RAGTurtorial.ipynb`, `notebooks/cppChat.ipynb` | Walkthroughs for loading, splitting, embedding, and searching documents. |
| Neural-network notebooks | `notebooks/GAN.ipynb`, `notebooks/mini_gpt.ipynb` | TensorFlow GAN and small GPT-style language-model examples. |

The edited Day 1 course notes are in [docs/notes/deep-learning-training-day-1.md](docs/notes/deep-learning-training-day-1.md). Research reading is in `papers/`.

## Setup and run

Use Python 3.10 or newer. Create a virtual environment for the demo you plan to run.

### College document assistant

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app uses local TF-IDF retrieval and does not call a hosted model. It uses the five files in `data/sample_documents/` unless documents are uploaded in the sidebar. Uploaded files are written locally to `.uploaded_documents/`.

### CNN image classifier

In a separate virtual environment:

```bash
python -m pip install -r requirements-cnn.txt
streamlit run CNNapp.py
```

The first CIFAR-10 training run may download the dataset. For custom training, upload a ZIP with one directory per class, containing that class's images.

### LangChain / Ollama C++ demos

```bash
python -m pip install -r requirements-llm.txt
streamlit run chatBotDemo.py
```

These demos use `data/reference/cpp.txt`. `ragChatbotLLM.py` and `cpp_tutorial_LLM.py` also need Ollama running with the `llama3.1:8b` model available. The embedding model may be downloaded on first use. These are exploratory examples; `app.py` is the maintained deterministic RAG example.

### Notebooks

Open notebooks in `notebooks/` with Jupyter or VS Code. `requirements-ml-notebooks.txt` provides TensorFlow, plotting, and notebook dependencies for the GAN and mini-GPT examples. For the LangChain RAG notebooks, install `requirements-llm.txt` and, for PDF support, `requirements.txt`. Check file paths in each notebook before running its cells. Notebook outputs and model training may require substantial memory and time.

## Main RAG design

The college app is split between `app.py` (Streamlit interface) and `rag.py` (retrieval and answer logic):

1. Documents are read and split into chunks with source names and, for PDFs, page numbers.
2. TF-IDF unigram and bigram vectors rank chunks by cosine similarity.
3. Answers select sentences from relevant passages; below a relevance threshold the app returns a not-found response.
4. Similar high-ranked passages from different files can trigger a conflict warning.
5. The interface displays supporting passages and their source metadata.

This small local demonstration uses lexical search rather than semantic search, and extractive sentence selection rather than LLM-generated answers.

## Repository hygiene

Keep virtual environments, Python caches, notebook checkpoints, local uploads, and generated model files out of Git. Sample source documents belong in `data/sample_documents/`; reference text belongs in `data/reference/`; research PDFs belong in `papers/`.
or the demo you plan to run.

## Projects

| Project | Entry point | What it does |
|---|---|---|
| College document assistant | `app.py` | Streamlit chat app that searches the included sample documents and uploaded PDF, Markdown, or text documents. Answers are extractive and show supporting passages; low-relevance questions are declined and disagreements between sources are flagged. |
| CNN image classifier | `CNNapp.py` | Trains a small CNN on CIFAR-10 or image folders supplied in a ZIP, then predicts a class for an uploaded image. Model construction is in `cnn_model.py`. |
| C++ RAG experiments | `chatBotDemo.py`, `ragChatbotLLM.py`, `cpp_tutorial_LLM.py` | Earlier experiments using LangChain, FAISS, sentence-transformer embeddings, and (in two scripts) Ollama. |
| Streamlit tutorial | `streamlit_tutorial.py` | Demonstrates common Streamlit inputs, charts, forms, and file upload. |
| RAG notebooks | `RAGTurtorial.ipynb`, `cppChat.ipynb` | Notebook walkthroughs of loading, splitting, embedding, and searching documents. |
| Neural network notebooks | `GAN.ipynb`, `mini_gpt.ipynb` | TensorFlow examples for a GAN and a small GPT-style language model. |

The included college sample documents are in `sample_documents/`. The C++ demos expect `cpp.txt` in the project root; make sure that file is available wherever you run them.

## Requirements and setup

Use Python 3.10 or newer. Create a virtual environment for the project you want to run so its packages do not interfere with other Python projects.

### College document assistant

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app uses local TF-IDF retrieval and does not call a hosted model. It uses the five sample files in `sample_documents/` unless documents are uploaded in the sidebar. Uploaded files are written to `.uploaded_documents/` locally.

Run the RAG checks with:

```bash
python -m pytest -q
```

### CNN image classifier

In a separate virtual environment, install:

```bash
python -m pip install -r requirements-cnn.txt
streamlit run CNNapp.py
```

The first CIFAR-10 training run may download the dataset. For custom training, upload a ZIP with one directory per class, containing that class's images.

### Earlier LangChain / Ollama C++ demos

Install their optional packages with:

```bash
python -m pip install -r requirements-llm.txt
```

Then run one of the scripts, for example:

```bash
streamlit run chatBotDemo.py
```

These demos require `cpp.txt`. `ragChatbotLLM.py` and `cpp_tutorial_LLM.py` also require Ollama to be installed and running, with the `llama3.1:8b` model available. The embedding model may be downloaded on first use. These are exploratory examples; the college assistant in `app.py` is the maintained, deterministic RAG example.

### Notebooks

Open notebooks with Jupyter or VS Code. `requirements-ml-notebooks.txt` provides the TensorFlow, plotting, and notebook dependencies used by the GAN and mini-GPT notebooks. For the LangChain RAG notebooks, install `requirements-llm.txt` as well as `requirements.txt` (which includes PDF support).

Notebook outputs and model training can require substantial memory and runtime. Run cells in order, and confirm file paths in the first loading cells match the files available on your computer.

## College RAG design

The main app is split between `app.py` (Streamlit UI) and `rag.py` (retrieval and answer logic):

1. Documents are read and split into chunks with source names and, for PDFs, page numbers.
2. TF-IDF unigram and bigram vectors rank chunks by cosine similarity.
3. Answers select sentences from relevant passages; below a relevance threshold the app returns a not-found response.
4. Similar, high-ranked passages from different files can trigger a conflict warning.
5. The UI displays the supporting passages and their source metadata.

This design is intended for a small local demonstration. TF-IDF is lexical search rather than semantic search, and the sentence selection is extractive rather than generated by an LLM.

## Repository hygiene

Keep virtual environments, Python caches, notebook checkpoints, local uploads, and generated model files out of Git. The `.gitignore` covers common local artifacts. The sample documents are app fixtures and should remain in `sample_documents/`.
