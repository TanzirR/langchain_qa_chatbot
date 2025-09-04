from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_rag_chain(retriever, llm):
    """
    Creates the complete conversational RAG chain.
    """

    # --- Prompt for rewriting the user's question ---
    # First, it sets up a prompt (contextualize_q_prompt) that handles follow-up questions.
    # Its only job is to look at the chat history and the user's latest input (e.g., "what about in 2008")
    # and rephrase it into a complete, standalone question (e.g., "What was the net interest cost in 2008?").
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
    # The create_history_aware_retriever takes that rephrased, standalone question and uses it to perform the semantic search on the PDF.
    # It finds the most relevant text chunks and passes them on to the next step.
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
             "If you don't know the answer, just say you don't know. DO NOT try to make up an answer. \n"
             "Do NOT use any outside knowledge.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # --- Create document processing chain ---
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # --- Combine all the pieces into the final RAG chain ---
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)