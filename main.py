import os
import sys
import re
import fitz 
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document


# --- Environment Variable Setup ---
# Load environment variables from a .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

# -------------------------------
# 1. PDF Processing → FAISS Vector Store
# -------------------------------

class FooterTextSplitter(RecursiveCharacterTextSplitter):
    # This class splits page text into chunks based on the footer
    # RecursiveCharacterTextSplitter is then applied to the pages, and if the footer is not present, it is injected into each chunk.
    # This will ensure the footer is present in each chunk to extract page no. as a source 
    # Specific ONLY for the financial_policy.pdf
    def __init__(self, footer_pattern, **kwargs):
        super().__init__(**kwargs)
        self.footer_pattern = re.compile(footer_pattern)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        final_chunks = []
        for doc in documents:
            # Find the footer once for the entire page 
            footer_match = self.footer_pattern.search(doc.page_content)
            footer_text = footer_match.group(0) if footer_match else ""

            # Perform the standard recursive split on the entire page content
            chunks_from_doc = super().split_text(doc.page_content)

            for chunk_text in chunks_from_doc:
                # Inject footer if it's not already present
                if footer_text and footer_text not in chunk_text:
                    chunk_text = f"{chunk_text}\n\n{footer_text}"
                
                new_metadata = doc.metadata.copy()
                final_chunks.append(Document(page_content=chunk_text, metadata=new_metadata))

        return final_chunks

def load_pdf_with_pymupdf(file_path):
    """
    Loads a PDF using PyMuPDF (fitz), extracting text from each page.
    """
    docs = []
    with fitz.open(file_path) as pdf_document:
        for page_num, page in enumerate(pdf_document):
            text = page.get_text()
            if text:  # Only add pages with actual text
                docs.append(Document(
                    page_content=text,
                    metadata={'page_number': page_num + 1}
                ))
    return docs

def prepare_pdf_retriever(file_path, api_key):
    """
    Loads a PDF, creates embeddings for each page, and returns a retriever.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at the specified path: {file_path}")

    docs = load_pdf_with_pymupdf(file_path)
    print(f" Loaded {len(docs)} pages from the PDF using PyMuPDF.")

    
    footer_pattern = r"2005-06 Budget Paper No\. 3\s+\d+" 
    text_splitter = FooterTextSplitter(
        footer_pattern=footer_pattern,
        chunk_size=1000, # I have kept the chunk size to 1000 and overlap to 100 to retain tables in the PDF properly
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)
    print(f" Split the document into {len(chunks)} chunks.")


    # Semantic Search (FAISS & OpenAIEmbeddings) for relevant chunk retrieval
    # It finds text chunks that are thematically similar, even if the exact same words are not used
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        model="text-embedding-3-large" # This embedding model is simpler to set up and works well with financial docs
    )

    db_faiss = FAISS.from_documents(chunks, embeddings)
    print(" Created FAISS vector store.")

    return db_faiss.as_retriever(search_kwargs={"k": 5}) # Returns the top 5 most relevant chunks.
                                                         # Provides enough context for the AI to form a complete answer.

# -------------------------------
# 2. Create the Conversational RAG Chain
# -------------------------------
def create_rag_chain(retriever, llm):
    """
    Creates the complete conversational RAG chain.
    """
    # --- Prompt for rewriting the user's question ---

    #First, it sets up a prompt (contextualize_q_prompt) that handles follow-up questions. 
    #Its only job is to look at the chat history and the user's latest input (e.g., "what about in 2008") 
    #and rephrase it into a complete, standalone question (e.g., "What was the net interest cost in 2008?").
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, \
             formulate a standalone question which can be understood without the chat history. \
             Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # --- Create a history-aware retriever ---

    #The create_history_aware_retriever takes that rephrased, standalone question and uses it to perform the semantic search on the PDF. 
    #It finds the most relevant text chunks and passes them on to the next step.
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # --- Prompt for answering the question ---
    
    # This step sets up the final prompt (qa_prompt) that instructs the AI on how to behave. 
    # It's told to act as a helpful assistant and to use only the text chunks provided by the retriever to form its answer. 
    # More importantly, I have mentioned to provide sources such as page number and if applicable, the table as well. 
    
    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful assistant for question-answering tasks. Use ONLY the retrieved context to answer. \n"
         "Special rule for this document: The page number is the digit that immediately follows the text "
         "'2005-06 Budget Paper No. 3'. \n"
         "- If the answer is found in a chunk containing this text, you MUST state the page number in your response as 'Source: Page <number>'. \n"
         "- If the answer is taken from a table, you MUST also mention 'table' after the page number, as 'Source: page <number>, Table <number>'. \n"
         "Do NOT use any outside knowledge.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

    # --- Create document processing chain ---
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # --- Combine all the pieces into the final RAG chain ---
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# -------------------------------
# 3. Handle a single question
# -------------------------------
def ask_question(query, chain, chat_history):
    """
    Asks a question to the RAG chain, prints the answer, and updates the chat history.
    """
    response = chain.invoke({
        "input": query,
        "chat_history": chat_history
    })

    answer = response.get("answer", "Sorry, I couldn't find an answer.")
    print(f"\n Answer: {answer}")
    
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))

# -------------------------------
# 4. Main execution block
# -------------------------------
def main():
    # --- Check for command-line argument for the PDF file ---
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]

    print("--- Initializing Chatbot ---")
    try:
        retriever = prepare_pdf_retriever(pdf_path, api_key=api_key)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # --- Initialize the language model ---
    llm = ChatOpenAI(
        temperature=0.0, # temperature can be slightly increased for creativity in answering questions.
        openai_api_key=api_key,
        model_name="gpt-4o-mini"
    )

    # --- Create the main RAG chain ---
    conversational_rag_chain = create_rag_chain(retriever, llm)
    
    # --- Initialize chat history ---
    chat_history = []

    print("--- Chatbot is ready! ---")
    print("Type your question and press Enter. CTRL + C to exit.")

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

