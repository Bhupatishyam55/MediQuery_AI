import streamlit as st
import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import PyPDF2
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai

# --------------------- CONFIG -------------------------
genai.configure(api_key="AIzaSyC4Rqb-nqgOBs73XtRehHsmr0aHlmQKEDE")
llm = genai.GenerativeModel("gemini-2.5-flash")

# Preload NLTK Resources
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------- FUNCTIONS ----------------------

def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def clean_text_nltk(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)

    cleaned = [
        word.lower()
        for word in tokens
        if word.isalpha() and word.lower() not in stop_words
    ]

    return " ".join(cleaned)


def make_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_text(text)


def build_faiss_index(chunks):
    embeddings = embed_model.encode(chunks).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def rewrite_query(query):
    prompt = f"""
    Rewrite this query to be medically precise and clear for document retrieval.
    Query: {query}

    Improved Query:
    """
    return llm.generate_content(prompt).text.strip()


def retrieve(query, index, chunks, k=3):
    improved = rewrite_query(query)
    query_vec = embed_model.encode([improved]).astype("float32")
    distances, indices = index.search(query_vec, k)
    return "\n\n".join([chunks[i] for i in indices[0]])


def answer(query, index, chunks):
    context = retrieve(query, index, chunks)

    prompt = f"""
    You are a professional medical report assistant.
    Use ONLY the below context to answer the user's question.
    Explain clearly and provide remedies/next steps in ONE LINE at the end.

    -------------------------
    CONTEXT:
    {context}
    -------------------------

    QUESTION: {query}
    """

    response = llm.generate_content(prompt)
    return response.text


# --------------------- STREAMLIT UI ----------------------

st.title("📄 Medical Report RAG Assistant")
st.write("Upload your medical PDF and ask questions about it.")

uploaded_pdf = st.file_uploader("Upload a Medical Report (PDF)", type=["pdf"])

if uploaded_pdf:
    st.success("PDF uploaded successfully!")

    if st.button("Process PDF"):
        with st.spinner("Extracting and indexing..."):
            raw_text = extract_text_from_pdf(uploaded_pdf)
            cleaned_text = clean_text_nltk(raw_text)
            chunks = make_chunks(cleaned_text)
            index = build_faiss_index(chunks)

        st.success("RAG System Ready!")
        st.session_state["chunks"] = chunks
        st.session_state["index"] = index

# -------------------- QUERY SECTION ----------------------

if "index" in st.session_state:

    query = st.text_input("Ask a question about your medical report")

    if st.button("Get Answer") and query.strip() != "":
        with st.spinner("Analyzing..."):
            response = answer(
                query,
                st.session_state["index"],
                st.session_state["chunks"]
            )
        st.subheader("💡 Answer")
        st.write(response)
