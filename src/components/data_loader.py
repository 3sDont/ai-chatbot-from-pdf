# src/components/data_loader.py
import streamlit as st
import fitz  # PyMuPDF
import io

class DataLoader:
    def load_from_path(self, file_path: str) -> str:
        """Đọc nội dung từ một file Markdown trên đĩa."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Lỗi khi đọc file từ đường dẫn {file_path}: {e}")
            return ""

    def load_from_upload(self, uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
        """Đọc nội dung text từ một file PDF được upload trực tiếp."""
        try:
            # Đọc file từ bộ nhớ
            file_bytes = uploaded_file.read()
            text = ""
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
            # Đặt lại con trỏ file về đầu để có thể đọc lại nếu cần
            uploaded_file.seek(0)
            return text
        except Exception as e:
            print(f"Lỗi khi đọc file PDF được upload: {e}")
            return ""
