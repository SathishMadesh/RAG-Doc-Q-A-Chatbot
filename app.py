import os
import streamlit as st
from langchain_groq import ChatGroq
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import tempfile


from dotenv import load_dotenv
load_dotenv()

## load the GROQ API Key
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

## load Google API key
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")


llm = ChatGroq(model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provide context only.
    Please provide the most accurate responce based on the question
    <context>
    {context}
    <context>
    Question:{input}
    """
)

def create_vector_embedding(uploaded_files):
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings(
            model="nomic-embed-text"
            )
        documents=[]
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path=tmp_file.name
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            documents.extend(docs)

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        final_docs = text_splitter.split_documents(documents)

        # Create vector store
        st.session_state.vectors = FAISS.from_documents(
            final_docs,
            st.session_state.embeddings
        )

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Document Embedding") and uploaded_files:
    with st.spinner("Processing documents..."):
        create_vector_embedding(uploaded_files)
    st.write("Vector Database is ready")

user_prompt=st.text_input("Enter your query from the research paper")

if st.button("Clear Database"):
    st.session_state.clear()

import time

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please upload and process documents first!")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        print(f"Response time: {time.process_time() - start}")

        st.write(response['answer'])

        with st.expander("Document similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('-----------------------')
    

