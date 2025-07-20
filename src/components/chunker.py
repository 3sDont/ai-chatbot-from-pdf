# src/components/chunker.py

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from typing import List

class Chunker:
    def __init__(self, chunk_size=1500, chunk_overlap=200):
        self.headers_to_split_on = [
            ("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on, strip_headers=False
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, document_text: str) -> List[str]:
        """Chia văn bản (ưu tiên cấu trúc Markdown nếu có)."""
        # Thử chia theo cấu trúc Markdown trước
        fragments = self.markdown_splitter.split_text(document_text)
        
        # Nếu chia theo Markdown không hiệu quả (chỉ ra 1 fragment lớn) thì dùng cách thông thường
        if len(fragments) <= 1:
            return self.text_splitter.split_text(document_text)

        # Nếu chia theo Markdown hiệu quả, tiếp tục chia nhỏ các fragment lớn
        chunks = self.text_splitter.split_documents(fragments)
        return [chunk.page_content for chunk in chunks]
