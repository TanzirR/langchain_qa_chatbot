from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def create_faiss_index(chunks: list, embeddings: OpenAIEmbeddings) -> FAISS:
    """
    Creates a FAISS vector store from document chunks.
    """
    return FAISS.from_documents(chunks, embeddings)