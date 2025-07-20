# src/components/chunker.py

from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from typing import List

class Chunker:
    def __init__(self, chunk_size=1000, chunk_overlap=150):
        # Chúng ta sẽ dùng splitter dành riêng cho Markdown
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, markdown_text: str) -> List[str]:
        """Chia văn bản Markdown theo cấu trúc tiêu đề trước, sau đó chia nhỏ các đoạn dài."""
        fragments = self.markdown_splitter.split_text(markdown_text)
        chunks = self.text_splitter.split_documents(fragments)
        # Chuyển đổi lại thành list of strings
        return [chunk.page_content for chunk in chunks]
