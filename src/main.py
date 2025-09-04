import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from pdf_processing import load_pdf_with_pymupdf, split_pdf_into_chunks
from retriever import create_pdf_retriever
from rag_chain import create_rag_chain
from chat import ask_question

# --- Environment Variable Setup ---
# Load environment variables from a .env file
def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

    # --- Check for command-line argument for the PDF file ---
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)
    pdf_path = sys.argv[1]

    print("--- Initializing Chatbot ---")
    if not os.path.exists(pdf_path):
        print(f"PDF not found at the specified path: {pdf_path}")
        sys.exit(1)

    # -------------------------------
    # 1. PDF Processing → FAISS Vector Store
    # -------------------------------
    docs = load_pdf_with_pymupdf(pdf_path)
    print(f" Loaded {len(docs)} pages from the PDF using PyMuPDF.")
    chunks = split_pdf_into_chunks(docs)
    print(f" Split the document into {len(chunks)} chunks.")
    retriever = create_pdf_retriever(chunks, api_key=api_key)
    print(" Created FAISS vector store.")

    # --- Initialize the language model ---
    llm = ChatOpenAI(
        temperature=0.0, # temperature can be slightly increased for creativity in answering questions.
        openai_api_key=api_key,
        model_name="gpt-4o-mini"
    )

    # -------------------------------
    # 2. Create the Conversational RAG Chain
    # -------------------------------
    conversational_rag_chain = create_rag_chain(retriever, llm)

    # --- Initialize chat history ---
    chat_history = []

    print("--- Chatbot is ready! ---")
    print("Type your question and press Enter. CTRL + C to exit.")

    # -------------------------------
    # 3. Handle a single question
    # -------------------------------
    # --- Interactive chat loop ---
    while True:
        try:
            query = input("\n Your question: ")
            ask_question(query, conversational_rag_chain, chat_history)
        except (KeyboardInterrupt, EOFError):
            print("\n Goodbye!")
            break

if __name__ == "__main__":
    main()

