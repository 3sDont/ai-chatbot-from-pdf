# text_splitter.py

class TextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        # Lưu ý: chunk_size và chunk_overlap ở đây tính theo số từ
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list:
        """
        Chia văn bản thành các đoạn (chunk) có độ dài chunk_size, với một phần chồng lặp (overlap)
        """
        if not isinstance(text, str):
            return []
            
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end >= len(words):
                break
            # Di chuyển có chồng lặp
            start += self.chunk_size - self.chunk_overlap
        return chunks
