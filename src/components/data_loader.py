# src/components/data_loader.py
import streamlit as st
import fitz  # PyMuPDF

class DataLoader:
    def load_from_upload(self, uploaded_file) -> str:
        """Đọc nội dung text từ một file được upload (PDF hoặc MD)."""
        file_type = uploaded_file.type
        
        try:
            file_bytes = uploaded_file.read()
            # Đặt lại con trỏ file về đầu để có thể đọc lại nếu cần
            uploaded_file.seek(0)
            
            if "pdf" in file_type:
                text = ""
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    for page in doc:
                        text += page.get_text()
                return text
            elif "markdown" in file_type or "text" in file_type:
                return file_bytes.decode("utf-8")
            else:
                return f"Lỗi: Định dạng file '{file_type}' không được hỗ trợ."
                
        except Exception as e:
            print(f"Lỗi khi đọc file được upload: {e}")
            return ""
