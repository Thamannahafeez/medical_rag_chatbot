
# app.py

import streamlit as st
import os
import pickle
from typing import List, Dict, Tuple
import re
import pandas as pd
import numpy as np
import torch

# Embedding & vector store
from sentence_transformers import SentenceTransformer

# FAISS
import faiss

# PDF reading
from PyPDF2 import PdfReader

# NLTK for text chunking
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Langchain components for LLM interaction
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# Assuming one of these LLM implementations will be used
from langchain_huggingface import HuggingFaceEndpoint
from transformers import pipeline, AutoTokenizer # Used in the notebook for HuggingFace pipeline

# free-up GPU
import torch, gc
gc.collect()
torch.cuda.empty_cache()


# Configurable defaults
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# EMBED_DIM = 384 # This should ideally be dynamically set based on the model

# EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-large-v2"
# EMBED_DIM = 1024

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIM = 768


# --- RAG System Components ---
# Embeddings
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
try:
    test_vec = embedder.encode("test")
    EMBED_DIM = len(test_vec)
except Exception:
    pass

def embed_texts(texts):
    vecs = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return np.array(vecs, dtype="float32")

# FAISS index helpers
def load_faiss_index(index_path):
    idx = faiss.read_index(index_path)
    return idx

# Persistence for metadata
def load_metadata(path):
    df = pd.read_csv(path)
    return df.to_dict('records')

# Retriever
def retrieve(index, query, k, metadata):
    qvec = embed_texts([query])
    faiss.normalize_L2(qvec)
    D, I = index.search(qvec, k)
    results = []
    text = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        results.append((float(score), metadata[idx]))
        text.append(metadata[idx]['chunk'])
    context = " ".join(text)
    return results, context

# Context assembly
def assemble_context(retrieved: List[Tuple[float, Dict]], max_chars: int = 2000) -> str:
    retrieved = sorted(retrieved, key=lambda x: x[0], reverse=True)
    parts = []
    cur = 0
    for score, md in retrieved:
        chunk = md.get("chunk", "")
        if cur + len(chunk) > max_chars:
            break
        parts.append(chunk)
        cur += len(chunk)
    return "\n\n---\n\n ".join(parts)

# LLM Initialization (Using HuggingFace pipeline)
def initialize_llm(model_id="google/flan-t5-base"):
    try:
        llm = pipeline(
            "text2text-generation",
            model=model_id,
            tokenizer=AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=True),
            device=0 if torch.cuda.is_available() else -1,
            max_length=256
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None
    
# Integrated Query Answering Function
def answer_query(index, metadata, query, llm, k=3, memory_window = 3):
    
    retrieved, content = retrieve(index, query, k=k, metadata=metadata)

    context = assemble_context(retrieved)

    # --- Conversation memory ---
    history = ""
    if "messages" in st.session_state:
        # Keep only last N exchanges
        past_turns = st.session_state["messages"][-(memory_window*2):]
        for role, text in past_turns:
            if role == "user":
                history += f"User: {text}\n"
            else:
                history += f"Assistant: {text}\n"

    template = """
        You are an assistant for a medical textbook. 

        Your task is to answer the user's question,
        in a simple, clear and easier to understand complete sentences
        by considering only the provided context and also the conversation history.
        Conversation history is provided to give you context of previous interactions.
        
        Converesation history:
        {history}
        Question: 
        {query}
        Context : 
        {context}

        If the answer is not in the context, **DO NOT** fabricate an answer,
        say "I don't know, Please consult a clinician."
        """
    system_prompt = PromptTemplate(
        input_variables = ['history','query','content'],
        template = template
        )
    
    formatted_prompt = system_prompt.format(history = history, query=query, context=content)
    
    answer = llm(formatted_prompt, do_sample=True, temperature=0.7, truncation=True,)[0]['generated_text']


    return answer   


# --- Streamlit Application Layout ---

st.title("Medical RAG Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [] 

index_path = "data/index.faiss"
meta_path = "data/meta.csv"

# Load index and metadata
try:
    # Attempt to load to verify paths are valid
    index = load_faiss_index(index_path)
    metadata = load_metadata(meta_path)
    #st.success()
    st.session_state['index_loaded'] = True
except Exception as e:
    st.error(f"Error loading index or metadata: {e}")
    st.stop()

# LLM Initialization
llm = initialize_llm()



# Display past messages
for role, text in st.session_state["messages"]:
    with st.chat_message(role):
        st.markdown(text)

# Chat Section
if prompt := st.chat_input("Ask a medical question..."):
    # Save user message
    st.session_state["messages"].append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(f"<div style='text-align: right'>{prompt}</div>", unsafe_allow_html=True)
        #st.markdown(prompt)

    # Bot response
    with st.chat_message("assistant"):
        with st.spinner("..."):
            result = answer_query(index, metadata, prompt, llm)
            st.markdown(result)

    # Save bot message
    st.session_state["messages"].append(("assistant", result))


