import os
import sys
from dotenv import load_dotenv

def load_api_key(env_var="OPENAI_API_KEY"):
    """
    Load the API key from environment variables.
    Raises an exception if the key is not found.
    """
    load_dotenv()
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"{env_var} not found in environment variables. Please set it in your .env file.")
    return api_key

def validate_pdf_path():
    """
    Check if a PDF path is provided in command-line arguments and exists.
    Returns the validated path.
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"PDF not found at the specified path: {pdf_path}")
        sys.exit(1)
      
    return pdf_path
