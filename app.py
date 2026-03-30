from dotenv import load_dotenv
import streamlit as st
import os

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# -----------------------------
# Load API Keys
# -----------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# -----------------------------
# Load LLM
# -----------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

# -----------------------------
# Extract text from PDFs
# -----------------------------
def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:
        reader = PdfReader(pdf)

        for page in reader.pages:
            text += page.extract_text() or ""

    return text


# -----------------------------
# Split text
# -----------------------------
def get_text_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_text(text)


# -----------------------------
# Create vector database
# -----------------------------
def create_vector_store(text_chunks):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    vectorstore = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    return vectorstore


# -----------------------------
# Ask question
# -----------------------------
def ask_question(vectorstore, question):

    docs = vectorstore.similarity_search(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the following context.

Context:
{context}

Question:
{question}

Answer clearly and in human readable format.
"""
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "context": context,
        "question": question
    })


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PDF Chatbot")

st.header("📄 Chat with PDFs")

pdf_docs = st.file_uploader(
    "Upload PDFs",
    accept_multiple_files=True
)

question = st.text_input("Ask a question about the documents")

process = st.button("Process Documents")

# -----------------------------
# Process PDFs
# -----------------------------
if process:

    if not pdf_docs:
        st.warning("Please upload at least one PDF.")
    else:

        with st.spinner("Reading PDFs..."):
            raw_text = get_pdf_text(pdf_docs)

        with st.spinner("Creating chunks..."):
            text_chunks = get_text_chunks(raw_text)

        with st.spinner("Creating vector database..."):
            vectorstore = create_vector_store(text_chunks)
            st.session_state.vectorstore = vectorstore

        st.success("Documents processed successfully!")


# -----------------------------
# Ask question
# -----------------------------
if question:

    if "vectorstore" not in st.session_state:
        st.warning("Please process PDFs first.")

    else:

        with st.spinner("Thinking..."):
            answer = ask_question(
                st.session_state.vectorstore,
                question
            )

        st.subheader("Answer")
        st.write(answer)