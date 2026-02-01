import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="AI Chatbot")

st.title("💬 AI Document Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    st.success("✅ PDF uploaded and processed!")

    query = st.chat_input("Ask from your PDF...")

    if query:
        st.chat_message("user").write(query)

        docs = retriever.invoke(query)

        answer = docs[0].page_content

        st.chat_message("assistant").write(answer)
