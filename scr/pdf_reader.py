# src/pdf_reader.py
import fitz  # PyMuPDF
import io

class PDFReader:
    def read(self, uploaded_file):
        text = ""
        try:
            # Mở file PDF từ dữ liệu bytes
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in pdf_document:
                text += page.get_text()
            pdf_document.close()
        except Exception as e:
            return f"Lỗi khi đọc file PDF: {e}"
        return text
