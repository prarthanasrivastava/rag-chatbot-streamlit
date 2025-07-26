# 🤖 RAG Chatbot with Streamlit & FLAN-T5

A Retrieval-Augmented Generation (RAG) chatbot built using Streamlit and Google's FLAN-T5. Upload your documents and ask natural language questions — the bot reads your data and answers contextually.

> 🔥 Live Demo: [https://rag-chatbot-streamlit.streamlit.app](https://rag-chatbot-streamlit.streamlit.app)

---

## 📂 Features

- 📄 Upload multiple files: PDF, DOCX, or TXT
- 🔍 Extracts and chunks document content
- 🧠 Embeds chunks using `sentence-transformers`
- 💬 Answers questions using `FLAN-T5` via `transformers`
- 📜 Displays complete chat history
- 💾 Download chat as PDF
- 🎨 Theme toggle: light/dark
- 📊 Sidebar document preview
- ✅ Deployed on Streamlit Cloud

---

## 🧠 Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | [Streamlit](https://streamlit.io) |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Model | `google/flan-t5-base` |
| PDF/Doc Parser | `PyPDF2`, `python-docx` |
| Vector DB | `FAISS` |
| Deployment | Streamlit Cloud |

---

## 🚀 Getting Started Locally

### 🔧 1. Clone the Repo

```bash
git clone https://github.com/prarthanasrivastava/rag-chatbot-streamlit.git
cd rag-chatbot-streamlit
