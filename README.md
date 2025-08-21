# LangChain PDF QA Chatbot

## 📝 Description

This project is a conversational Question-Answering (QA) chatbot. It uses the LangChain framework and OpenAI's language models to read a local PDF document, understand its content, and answer questions about it in an interactive command-line interface.

The chatbot is designed to be "history-aware," meaning it can understand and answer follow-up questions based on the ongoing conversation. It also cites the page numbers from the source PDF where it found the information for its answers.

---

## ✨ Features

* **Interactive Chat**: Run the chatbot in your terminal and ask questions conversationally.
* **PDF Processing**: Automatically loads and processes text from any PDF file passed as a command-line argument.
* **Vector Store**: Creates a searchable vector index of the PDF's content using FAISS for efficient information retrieval.
* **Conversational Memory**: Remembers the chat history to understand context and answer follow-up questions.
* **Source Citation**: Identifies and displays the exact page numbers from the PDF that were used to generate an answer.
* **Modern LangChain**: Built using the latest LangChain Expression Language (LCEL) for robust and efficient chain construction.

---

## ⚙️ Setup and Installation

Follow these steps to get the chatbot running on your local machine.

### 1. Prerequisites

* Python 3.8 or higher
* An OpenAI API key

### 2. Clone the Repository (Optional)

If you have this project in a git repository, clone it first:

```bash
git clone https://github.com/TanzirR/langchain_qa_chatbot.git
cd langchain_qa_chatbot
```

### 3. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 4. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables

Create a file named `.env` in the root directory of the project. Add your OpenAI API key to this file. This keeps your key secure and out of the main script.

```
OPENAI_API_KEY="your_openai_api_key_here"
```

---

## 🚀 How to Run

Once you have completed the setup, you can run the script from your terminal.

1.  **Run the script** from your terminal, passing the path to your PDF file as a command-line argument:

    ```bash
    python main.py "path/to/your/document.pdf"
    ```

    For example:

    ```bash
    python main.py financial_policy.pdf
    ```

2.  **Start Chatting**: The script will process the PDF and display a prompt. You can now ask your questions.

    ```
    --- Chatbot is ready! ---
    Type your question and press Enter. Type 'quit' to exit.

    Your question: What is the total territory net asset in 2008?
    ```

4.  **Exit the Chatbot**: To end the session, simply type `quit` and press Enter.
