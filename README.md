# LangChain PDF QA Chatbot with FastAPI

## 📝 Description

This project is a conversational Question-Answering (QA) chatbot built with Python, FastAPI, and the LangChain framework. It allows you to upload a PDF document via a REST API, processes it, and answers questions about its content. The application is designed to be persistent, saving vector stores for each document and maintaining chat history for conversational context.

This allows for a robust, scalable solution where multiple documents can be queried independently, and conversations can be picked up later.

---

## ✨ Features

*   **RESTful API**: Built with FastAPI, providing endpoints to upload PDFs and ask questions.
*   **Asynchronous PDF Processing**: PDF ingestion and vectorization run as a background task, so the API remains responsive.
*   **Persistent Document Storage**: Creates and saves a FAISS vector store for each uploaded document, identified by a unique ID.
*   **Conversational Memory**: Maintains a history for each conversation, enabling context-aware follow-up questions.
*   **RAG Architecture**: Implements a Retrieval-Augmented Generation (RAG) chain using LangChain Expression Language (LCEL).
*   **Easy Configuration**: Manages API keys and settings through a `.env` file.

---

## 🏛️ Project Structure

```
.
├── .env                  # Environment variables (OpenAI API Key)
├── main.py               # FastAPI application entry point
├── requirements.txt      # Python dependencies
├── src/                  # Core application logic
│   ├── chat.py           # Chat interaction logic
│   ├── config.py         # Configuration settings
│   ├── pdf_processing.py # PDF loading and chunking
│   ├── rag_chain.py      # RAG chain creation
│   ├── retriever.py      # FAISS vector store and retriever logic
│   └── state_manager.py  # Handles saving/loading of state
├── data/                 # Example PDF documents
├── vector_stores/        # Directory for saved FAISS vector stores
└── chat_histories/       # Directory for saved chat histories
```

---

## ⚙️ Setup and Installation

### 1. Prerequisites

*   Python 3.10 or higher
*   An OpenAI API key

### 2. Clone this specific branch ```rag-fastapi```

```bash
git clone --branch rag-fastapi --single-branch https://github.com/TanzirR/langchain_qa_chatbot.git
cd langchain_qa_chatbot
```

### 3. Create a Virtual Environment

It's highly recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 4. Install Dependencies

Install the required Python packages.

```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables

Create a file named `.env` in the root directory and add your OpenAI API key.

```env
# .env
OPENAI_API_KEY="your_openai_api_key_here"
```

---

## 🚀 How to Run

1.  **Start the FastAPI Server**:
    Run the application using `uvicorn`.

    ```bash
    uvicorn main:app --reload
    ```

2.  **Access the API Documentation**:
    Once the server is running, navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to access the interactive Swagger UI documentation.

---

## 📖 API Usage

### 1: Upload a PDF Document

*   **Endpoint**: `POST /upload`
*   Use this endpoint to upload a PDF file. The server will process it in the background and create a vector store.
*   **Response**: You will receive a `document_id` which you must use for querying.

**Example using `curl`:**

```bash
curl -X POST "http://127.0.0.1:8000/upload-pdf" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "file=@/path/to/your/document.pdf"
```
**Expected Response:**
```json
{
  "message": "PDF upload accepted. Processing in the background.",
  "document_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef"
}
```

### 2: Query the Document

Now, use the document_id you received to ask questions. You send a ```POST``` request to the ```/query/{document_id}``` endpoint with your question in a JSON body. You can also specify a ```session_id``` to maintain a conversation history.

Using ```curl``` replace ```{document_id}``` with the ID from Step 1.

```bash
curl -X POST "http://127.0.0.1:8000/query/a1b2c3d4-e5f6-7890-1234-567890abcdef?session_id=my-first-chat" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{"query": "What is the main topic of this document?"}'
```
**Expected Response:**
```json
{
  "answer": "The main topic of the document appears to be about the annual financial performance of the company, detailing revenue, costs, and net profit."
}
```


