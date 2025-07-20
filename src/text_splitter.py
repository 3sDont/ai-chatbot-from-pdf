# src/text_splitter.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

class TextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, text: str) -> List[str]:
        """Chia văn bản thành các chunks."""
        return self.splitter.split_text(text)
