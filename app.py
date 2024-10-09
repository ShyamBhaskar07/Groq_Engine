import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']

# Caching the vector store to avoid reloading and processing each time
@st.cache_resource
def load_vector_store():
    embeddings = OllamaEmbeddings()
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Machine_learning")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:50])
    vector = FAISS.from_documents(final_documents, embeddings)
    return vector

# Button to load the vector store
if "vector" not in st.session_state:
    st.session_state.vector = None

if st.button("Load Vector Store") and st.session_state.vector is None:
    st.session_state.vector = load_vector_store()
    st.success("Vector Store Loaded Successfully!")

st.title("ChatGroq Demo")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

prompt = ChatPromptTemplate.from_template(
    """
    Please provide the most accurate answer according to context only
    <context>
    {context}
    <context>

    Questions: {input}
    """
)

if st.session_state.vector is not None:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    user_prompt = st.text_input("Enter your prompt")

    if user_prompt:
        response = retrieval_chain.invoke({"input": user_prompt})
        st.write(response['answer'])
else:
    st.warning("Please load the vector store before asking questions.")
