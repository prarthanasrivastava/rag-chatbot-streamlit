import os 

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from io import StringIO

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub

# 🔁 Load environment variables (including Hugging Face token)
load_dotenv()

st.set_page_config(page_title="📚 RAG Chatbot with HuggingFace", layout="wide")

def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        pdf = PdfReader(uploaded_file)
        for page in pdf.pages:
            text += page.extract_text() or ""
    elif uploaded_file.name.endswith(".docx"):
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif uploaded_file.name.endswith(".txt"):
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        text = stringio.read()
    return text

def main():
    st.title("🤖 Hugging Face RAG Chatbot")
    st.markdown("Upload a PDF, DOCX, or TXT file, and ask questions about it!")

    uploaded_file = st.file_uploader("📤 Drag and drop your file here", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        raw_text = extract_text_from_file(uploaded_file)

        if not raw_text.strip():
            st.warning("❌ The file appears empty or could not be read.")
            return

        # Split into chunks
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(raw_text)

        if not chunks:
            st.warning("❌ No text chunks were created from the file.")
            return

        # Vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        st.success("✅ File processed and indexed successfully!")

    # Chat input and logic
    query = st.text_input("💬 Ask a question about the uploaded file:")
    if query:
        if "vectorstore" not in st.session_state:
            st.warning("❗ Please upload and process a document first.")
            return

        docs = st.session_state.vectorstore.similarity_search(query)

        if not docs:
            st.info("🤔 Couldn't find relevant content. Try rephrasing your question.")
            return

        # 🔐 Use Hugging Face LLM with API token from .env
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            model_kwargs={"temperature": 0.3, "max_length": 300}
        )

        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)

        st.markdown("### 📜 Answer")
        st.success(response)

if __name__ == "__main__":
    main()
