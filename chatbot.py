import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.warning("Google API key not found. Please add it to a .env file in the project root as GOOGLE_API_KEY=your-key")

# Ensure an asyncio loop exists (Streamlit runs in a separate thread)
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

st.title("📄 PDF Q&A Chatbot (Google Gemini)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

# File uploader at the top
file = st.file_uploader("Upload a PDF", type="pdf")

if file is not None and not st.session_state.file_uploaded:
    # ---- Extract text from PDF ----
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # ---- Split into chunks ----
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    if not chunks:
        st.warning("No extractable text found in the PDF. Please upload a different file.")
    else:
        # ---- Embeddings ----
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

        # ---- Vector store ----
        vector_store = FAISS.from_texts(chunks, embeddings)
        st.session_state.vector_store = vector_store
        st.session_state.file_uploaded = True

# ---- LLM ----
llm = ChatGoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_output_tokens=1000,
    model="models/gemini-1.5-pro-latest"
)

# ---- Prompt ----
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the provided context to answer the question.
If you cannot find the answer, say you don't know.

Context:
{context}

Question:
{question}
""")

# ---- Chain ----
chain = create_stuff_documents_chain(llm, prompt)

# ---- Chat Interface ----
for msg in st.session_state.get("chat_history", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
user_input = st.chat_input("Type your question here...")

if user_input and st.session_state.vector_store:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Retrieve relevant context
    matches = st.session_state.vector_store.similarity_search(user_input)
    response = chain.invoke({
        "context": matches,
        "question": user_input
    })

    answer = response if isinstance(response, str) else str(response)
    # Add assistant message
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    st.rerun()
elif user_input and not st.session_state.vector_store:
    st.warning("Please upload a PDF before asking questions.")
