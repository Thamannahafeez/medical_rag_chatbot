# Medical RAG Chatbot

A **medical question-answering chatbot** leveraging Retrieval-Augmented Generation (RAG) for safe, context-aware responses using medical documents. Built with Streamlit, Hugging Face, FAISS, and PyTorch.

***

## Features

- Retrieval-Augmented Generation (RAG) for enhanced answer accuracy.
- Embedding medical documents using SentenceTransformers.
- Context-aware and memory-enabled conversation.
- Simple, interactive Streamlit web interface.
- Prioritizes user safety; does not answer out-of-context medical questions.

***

## Tech Stack

- **Python 3.8+**
- **Streamlit** for the UI
- **PyTorch** for deep learning
- **Sentence-Transformers** for text embeddings
- **FAISS** for vector search
- **LangChain** for prompt construction
- **transformers** for LLM pipelines
- **Other**: pandas, numpy, nltk, PyPDF2

***

## Getting Started

### Prerequisites

- Python 3.8 or above
- pip (Python package manager)
- Medical data files (`data/index.faiss`, `data/meta.csv`)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Thamannahafeez/medical-rag-chatbot.git
    cd medical-rag-chatbot
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Place your FAISS index and metadata files in the `data/` folder.

### Usage

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will be available at [http://localhost:8501](http://localhost:8501) by default.

***

## Project Structure

| File/Folder             | Description                                                 |
|-------------------------|-------------------------------------------------------------|
| `app.py`                | Main Streamlit application file                             |
| `requirements.txt`      | Python dependencies                                         |
| `data/index.faiss`      | FAISS embedding index                                       |
| `data/meta.csv`         | Chunk metadata for retrieval                                |

***

## Data

- You can supply a FAISS index file containing embeddings and a corresponding metadata CSV in the `data/` directory. This repo contains proprietary medical data embedded from a PDF version of a medical textbook named "Gupte-The-Short-Textbook-of-Pediatrics-11th-Ed-2009" .

***

## Limitations and Disclaimer

- The chatbot answers only using context from the uploaded documents. If the answer is not known or context is insufficient, it will refuse to respond or direct the user to a clinician.
- **Do not use for clinical diagnosis or decision-making.**

***


## Acknowledgments

- Based on open-source RAG chatbot architectures and inspired by similar medical FAQ bots
- Uses components from HuggingFace, SentenceTransformers, and FAISS.

***

