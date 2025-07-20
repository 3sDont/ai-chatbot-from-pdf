# src/text_splitter.py

def split_text(text, chunk_size=500, overlap=50):
    """
    Chia văn bản thành các đoạn (chunk) có độ dài chunk_size, với một phần chồng lặp (overlap)
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # di chuyển có chồng lặp
    return chunks
