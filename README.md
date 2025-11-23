# 📄 Medical Report RAG Assistant

A Streamlit application that allows users to upload medical PDFs and ask questions about them using a Retrieval-Augmented Generation (RAG) pipeline.  
The system extracts text, cleans it, chunks it, embeds it using Sentence Transformers, and retrieves relevant segments using FAISS.  
Google Gemini (2.5 Flash) is used for answering queries.

---

## 🚀 Features
- PDF text extraction (PyPDF2)
- NLTK text cleaning and stopword removal
- Semantic chunking using LangChain splitters
- FAISS vector database for retrieval
- Gemini 2.5 Flash for medical question answering
- Simple Streamlit UI

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/medical-report-rag-assistant
cd medical-report-rag-assistant
pip install -r requirements.txt
