import pdfplumber
from docx import Document
from typing import List, Union
import os

class DocumentProcessor:
    """Handles processing of different document formats and text chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size (int): The size of text chunks in characters
            chunk_overlap (int): The overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_document(self, file_path: str) -> str:
        """
        Process a document file and extract its text content.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            str: Extracted text content
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self._process_pdf(file_path)
        elif file_extension == '.docx':
            return self._process_docx(file_path)
        elif file_extension == '.txt':
            return self._process_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def _process_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        text_content = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text_content.append(page.extract_text())
        return '\n'.join(text_content)

    def _process_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        doc = Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

    def _process_text(self, file_path: str) -> str:
        """Read text from plain text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def create_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text (str): Input text to be chunked
            
        Returns:
            List[str]: List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk of text
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # If not at the end of text, try to find a good break point
            if end < len(text):
                # Try to break at the last period or newline
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point != -1:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
            
        return chunks

    def process_and_chunk(self, file_path: str) -> List[str]:
        """
        Process a document and split it into chunks.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            List[str]: List of text chunks
        """
        text = self.process_document(file_path)
        return self.create_chunks(text)
