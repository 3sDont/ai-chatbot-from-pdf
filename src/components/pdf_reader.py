# src/pdf_reader.py
import fitz  # PyMuPDF
import io

class PDFReader:
    def read(self, uploaded_file) -> str:
        """Đọc text từ file PDF được upload (đối tượng bytes)."""
        text = ""
        try:
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            print(f"Lỗi khi đọc file PDF: {e}")
            return ""  # Trả về chuỗi rỗng nếu có lỗi
        return text
