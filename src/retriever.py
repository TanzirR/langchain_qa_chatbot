from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def create_pdf_retriever(chunks, api_key, embedding_model="text-embedding-3-large", k=5):
    """
    Semantic Search (FAISS & OpenAIEmbeddings) for relevant chunk retrieval.
    It finds text chunks that are thematically similar, even if the exact same words are not used.
    Returns the top k most relevant chunks, providing enough context for the AI to form a complete answer.
    """
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        model=embedding_model # This embedding model is simpler to set up and works well with financial docs
    )
    db_faiss = FAISS.from_documents(chunks, embeddings)
    return db_faiss.as_retriever(search_kwargs={"k": k})