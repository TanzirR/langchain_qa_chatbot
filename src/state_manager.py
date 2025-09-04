import os
import json
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# --- Directory Setup ---
VECTOR_STORE_DIR = "vector_stores"
CHAT_HISTORY_DIR = "chat_histories"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# --- Vector Store Management ---
def get_vector_store_path(doc_id: str) -> str:
    return os.path.join(VECTOR_STORE_DIR, f"{doc_id}.faiss")

def save_vector_store(db: FAISS, doc_id: str):
    """Saves a FAISS vector store to disk."""
    db.save_local(get_vector_store_path(doc_id))

def load_vector_store(doc_id: str, embeddings: OpenAIEmbeddings) -> FAISS | None:
    """Loads a FAISS vector store from disk."""
    db_path = get_vector_store_path(doc_id)
    if not os.path.exists(db_path):
        return None
    # Note: allow_dangerous_deserialization is needed for FAISS.
    # Ensure you trust the source of the FAISS index file.
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

# --- Chat History Management ---
def get_chat_history_path(session_id: str) -> str:
    return os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")

def save_chat_history(session_id: str, history: list):
    """Saves a chat history to a JSON file."""
    with open(get_chat_history_path(session_id), 'w') as f:
        json.dump(history, f)

def load_chat_history(session_id: str) -> list:
    """Loads a chat history from a JSON file."""
    history_path = get_chat_history_path(session_id)
    if not os.path.exists(history_path):
        return []
    with open(history_path, 'r') as f:
        return json.load(f)