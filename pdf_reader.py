# pdf_reader.py

import fitz  # PyMuPDF
import io

class PDFReader:
    def read(self, uploaded_file):
        """
        Đọc nội dung text từ một file PDF đã được upload qua Streamlit.
        `uploaded_file` là một đối tượng giống file (file-like object).
        """
        text = ""
        try:
            # Mở file PDF từ dữ liệu bytes trong bộ nhớ
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in pdf_document:
                text += page.get_text()
            pdf_document.close()
        except Exception as e:
            st.error(f"Lỗi khi đọc file PDF: {e}")
            return None
        return text
