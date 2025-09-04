from langchain_core.messages import HumanMessage, AIMessage

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