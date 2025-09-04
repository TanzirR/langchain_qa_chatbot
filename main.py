import uuid
import os
from fastapi import FastAPI, UploadFile, File, Query, BackgroundTasks, HTTPException
from pydantic import BaseModel

# Corrected imports to reference the 'src' package
from src.pdf_processing import load_pdf_with_pymupdf, split_pdf_into_chunks
from src.retriever import create_faiss_index
from src.rag_chain import create_rag_chain
from src.config import settings
from src.state_manager import (
    save_vector_store,
    load_vector_store,
    save_chat_history,
    load_chat_history,
)

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

app = FastAPI(title="RAG API with Persistent State", version="2.0")

# --- Background Task for PDF Processing ---
def process_and_index_pdf(file_path: str, document_id: str):
    """
    Background task to process a PDF file and create a vector store.
    """
    try:
        print(f"Starting processing for document_id: {document_id}")
        docs = load_pdf_with_pymupdf(file_path)
        chunks = split_pdf_into_chunks(docs)

        embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model="text-embedding-3-large"
        )
        db_faiss = create_faiss_index(chunks, embeddings)
        save_vector_store(db_faiss, document_id)
        print(f"Successfully processed and indexed document_id: {document_id}")
    except Exception as e:
        print(f"Failed to process {document_id}. Error: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

# --- API Schemas ---
class UploadResponse(BaseModel):
    message: str
    document_id: str

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# --- API Endpoints ---
@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts a PDF file, saves it, and starts a background task for processing.
    Returns a document_id immediately.
    """
    document_id = str(uuid.uuid4())
    file_path = f"temp_{document_id}_{file.filename}"

    # Save uploaded file temporarily
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Start the processing in the background
    background_tasks.add_task(process_and_index_pdf, file_path, document_id)

    return UploadResponse(
        message="PDF upload accepted. Processing in the background.",
        document_id=document_id
    )

@app.post("/query/{document_id}", response_model=QueryResponse)
async def query_rag(document_id: str, request: QueryRequest, session_id: str = Query("default")):
    """
    Asks a question to a specific document, with conversation history support.
    """
    embeddings = OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,
        model="text-embedding-3-large"
    )
    
    # Load the specific vector store for this document
    vector_store = load_vector_store(document_id, embeddings)
    if vector_store is None:
        raise HTTPException(status_code=404, detail="Document not found or still processing.")

    # Create the RAG chain
    llm = ChatOpenAI(openai_api_key=settings.openai_api_key, model="gpt-4o-mini")
    retriever = vector_store.as_retriever()
    rag_chain = create_rag_chain(retriever, llm)

    # Get session history
    history = load_chat_history(session_id)

    # Run RAG with history
    try:
        result = rag_chain.invoke({"input": request.query, "chat_history": history})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during RAG chain invocation: {e}")

    # Update and save session history
    history.append(("human", request.query))
    history.append(("ai", result["answer"]))
    save_chat_history(session_id, history)

    return QueryResponse(answer=result["answer"])