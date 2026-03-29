# 🚀 Basic RAG (Retrieval Augmented Generation)

This project demonstrates a simple end-to-end implementation of a **RAG (Retrieval Augmented Generation)** pipeline using LangChain, Hugging Face embeddings, and FAISS.

---

## 📌 What is RAG?

RAG is an architecture that enhances LLM responses by combining:

* **Retrieval** → Fetch relevant information from external data
* **Augmentation** → Add retrieved context to the prompt
* **Generation** → Generate answer using LLM

---

## 🧠 Architecture Flow

1. Load document (`data.txt`)
2. Split into chunks
3. Convert chunks → embeddings using Hugging Face
4. Store embeddings in FAISS (vector database)
5. Retrieve relevant chunks based on query
6. Pass **context + query** to LLM
7. Generate final answer

---

## 📂 Project Structure

```
.
├── data.txt              # Input knowledge base
├── loadData.py           # Load + split documents
├── vector_store.py       # Create embeddings + FAISS DB
├── rag_pipeline.py       # End-to-end RAG pipeline
├── .gitignore            # Ignore sensitive files
```

---

## ⚙️ Tech Stack

* Python
* LangChain
* Hugging Face (Embeddings + Models)
* FAISS (Vector DB)

---

## ▶️ How to Run

1. Install dependencies:

```
pip install langchain langchain-community sentence-transformers faiss-cpu transformers
```

2. Run pipeline:

```
python rag_pipeline.py
```

