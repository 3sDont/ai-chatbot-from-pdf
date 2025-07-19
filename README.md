
# 📚 AI Chatbot từ PDF với Streamlit

Dự án này xây dựng một **AI chatbot** có khả năng trả lời các câu hỏi của người dùng dựa trên **nội dung của file PDF** (giáo trình, tài liệu học tập, v.v.), sử dụng các mô hình mã nguồn mở miễn phí.

## 🎯 Mục tiêu
Tạo một sản phẩm demo cho phép:
- Người dùng **upload file PDF**
- Chatbot **hiểu nội dung** trong tài liệu
- Người dùng có thể **đặt câu hỏi tự do**
- Trả lời dựa trên ngữ cảnh thực sự trong tài liệu

## 🧠 Công nghệ sử dụng
- `Streamlit` — Giao diện web đơn giản và trực quan
- `PyPDF2` / `pdfplumber` — Đọc nội dung PDF
- `LangChain` — Quản lý chain xử lý tài liệu và hỏi đáp
- `SentenceTransformers` — Embedding văn bản
- `Chroma` — Vector store để lưu embedding
- `HuggingFace Transformers` — Model trả lời câu hỏi (tùy chọn)

## ⚙️ Cấu trúc thư mục

```
AI-CHATBOT-FROM-PDF/
├── app.py                # Streamlit UI
├── requirements.txt      # Các thư viện cần cài đặt
├── README.md             # Mô tả dự án
└── src/
    ├── pdf_reader.py         # Đọc file PDF
    ├── embedders.py          # Tạo vector embedding
    ├── vector_store.py       # Lưu vector vào Chroma
    ├── rag_pipeline.py       # Tạo câu trả lời từ LLM
    ├── conversation.py       # Duy trì session chat
```

## 🚀 Hướng dẫn chạy local

1. Clone dự án về máy:
   ```bash
   git clone https://github.com/3sDont/AI-CHATBOT-FROM-PDF.git
   cd AI-CHATBOT-FROM-PDF
   ```

2. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```

3. Chạy ứng dụng:
   ```bash
   streamlit run app.py
   ```

## ☁️ Deploy online miễn phí

Bạn có thể deploy ứng dụng này lên [Streamlit Community Cloud](https://streamlit.io/cloud) để dùng online. Xem hướng dẫn chi tiết trong phần triển khai.

## 📎 Ví dụ sử dụng

1. Tải lên một giáo trình dạng PDF
2. Đặt câu hỏi như:
   - "Tóm tắt nội dung chương 3"
   - "Thuật toán Apriori dùng để làm gì?"
   - "Hàm softmax hoạt động như thế nào?"

## 📌 Lưu ý
- Ứng dụng này chạy tốt với tài liệu tiếng Việt hoặc tiếng Anh
- Mô hình LLM sử dụng là mô hình **nhẹ** để phù hợp với giới hạn của Streamlit Cloud
- Không phù hợp cho file quá lớn (> 50MB)

## 👨‍💻 Tác giả

- Trần Bá Đông – Đại học Khoa học Tự nhiên TP.HCM
- Contact: [tranbadong9471@gmail.com](mailto:tranbadong9471@gmail.com)

---

📢 *Hãy thử trải nghiệm tại:*  
👉 [https://3sDont-ai-chatbot-from-pdf.streamlit.app](https://3sDont-ai-chatbot-from-pdf.streamlit.app)
