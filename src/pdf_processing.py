import fitz
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

def split_pdf_into_chunks(docs, chunk_size=1000, chunk_overlap=100):
    """
    Splits documents into chunks.
    I have kept the chunk size to 1000 and overlap to 100 to retain tables in the PDF properly.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)