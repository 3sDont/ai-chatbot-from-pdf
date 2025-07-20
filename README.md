# 🤖 AI Chatbot: Trợ Lý Hỏi Đáp Thông Minh Từ Tài Liệu

Dự án này xây dựng một **AI Chatbot** mạnh mẽ, có khả năng "đọc hiểu" và trả lời các câu hỏi phức tạp dựa trên nội dung của nhiều loại tài liệu (PDF, Markdown), kể cả các tài liệu kỹ thuật chứa công thức, bảng biểu và mã nguồn.

Đây không chỉ là một ứng dụng demo, mà là một **pipeline xử lý tài liệu thông minh (Intelligent Document Processing)** hoàn chỉnh, được xây dựng theo kiến trúc module chuyên nghiệp.

👉 **Trải nghiệm ứng dụng tại:** [[3sdont-ai-chatbot](https://3sdont-ai-chatbot-from-pdf.streamlit.app/)]

## 🎯 Tính Năng Nổi Bật

- **Hỗ Trợ Đa Định Dạng:** Cho phép người dùng tải lên trực tiếp file `.pdf` hoặc `.md`.
- **Chất Lượng Trích Xuất Vượt Trội:** Tích hợp công cụ **Marker** để chuyển đổi PDF phức tạp thành Markdown có cấu trúc, giữ nguyên công thức toán, bảng biểu và khối code.
- **Hiểu Ngữ Nghĩa Sâu:** Sử dụng các mô hình embedding hàng đầu để vector hóa và tìm kiếm thông tin dựa trên ngữ nghĩa thay vì từ khóa.
- **Trả Lời Thông Minh & Siêu Nhanh:** Tận dụng sức mạnh của model **Llama 3** thông qua **Groq API**, mang lại câu trả lời chất lượng cao với tốc độ đáng kinh ngạc.
- **Kiến Trúc Module Chuyên Nghiệp:** Mã nguồn được tổ chức rõ ràng, dễ bảo trì, và mở rộng.

## 🧠 Công Nghệ Cốt Lõi

| Chức Năng | Công Nghệ | Lý Do Lựa Chọn |
| :--- | :--- | :--- |
| **Giao diện Web** | `Streamlit` | Nhanh chóng, trực quan, phù hợp cho các ứng dụng AI. |
| **Tiền xử lý PDF** | `Marker` / `PyMuPDF` | **Marker** cho chất lượng trích xuất tài liệu kỹ thuật tốt nhất. **PyMuPDF** cho xử lý nhanh online. |
| **Orchestration & Components** | `LangChain` | Cung cấp các công cụ mạnh mẽ (`TextSplitters`, `PromptTemplates`) để xây dựng pipeline RAG. |
| **Embedding Văn bản** | `SentenceTransformers` | Hiệu quả, mã nguồn mở, hoạt động tốt với cả tiếng Anh và tiếng Việt. |
| **LLM (Trái tim AI)** | **Llama 3** qua `Groq API` | Kết hợp giữa mô hình ngôn ngữ mã nguồn mở hàng đầu và tốc độ xử lý vượt trội của LPU. |
| **Vector Search** | `Numpy` / `Scikit-learn` | Giải pháp tìm kiếm tương đồng đơn giản, hiệu quả cho quy mô demo, không cần DB ngoài. |

## ⚙️ Kiến Trúc Dự Án

Dự án được cấu trúc theo các module chuyên biệt để tối đa hóa khả năng bảo trì và mở rộng:
AI-CHATBOT-FROM-PDF/
│
├── documents/ # Nơi chứa dữ liệu (PDF gốc và Markdown đã xử lý)
├── src/
│ ├── components/ # Các "viên gạch" xây dựng: Loader, Chunker, Embedder,...
│ └── pipelines/ # Nơi lắp ráp components thành một dây chuyền RAG hoàn chỉnh
│
├── app.py # Giao diện Streamlit (mỏng, nhẹ)
├── preprocess_pdf.py # Script độc lập, mạnh mẽ để chuyển PDF -> Markdown
├── requirements.txt # Danh sách các thư viện cần cài đặt
└── README.md # Chính là file này


## 🚀 Hướng Dẫn Chạy Local

### 1. Điều Kiện Tiên Quyết
- Python 3.9+
- Một API Key từ [GroqCloud](https://console.groq.com/keys) (có bậc miễn phí)

### 2. Cài Đặt

1.  Clone dự án về máy:
    ```bash
    git clone https://github.com/3sDont/AI-CHATBOT-FROM-PDF.git
    cd AI-CHATBOT-FROM-PDF
    ```

2.  Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Chạy Ứng Dụng

Có 2 chế độ để chạy ứng dụng:

**Chế độ A: Nhanh & Tiện Lợi (Chất lượng vừa phải)**

1.  Chạy trực tiếp ứng dụng Streamlit:
    ```bash
    streamlit run app.py
    ```
2.  Mở trình duyệt và truy cập vào địa chỉ local.
3.  Tải lên một file PDF bất kỳ và bắt đầu hỏi đáp.

**Chế độ B: Chất Lượng Cao (Đề xuất cho tài liệu phức tạp)**

1.  **Bước Tiền Xử Lý:**
    - Đặt các file PDF của bạn vào thư mục `documents/pdfs/`.
    - Chạy script tiền xử lý từ terminal:
      ```bash
      python preprocess_pdf.py documents/pdfs documents/markdowns
      ```
    - Script này sẽ tạo ra các file `.md` tương ứng trong thư mục `documents/markdowns/`.

2.  **Chạy Ứng Dụng:**
    - Chạy ứng dụng Streamlit:
      ```bash
      streamlit run app.py
      ```
    - Tải file `.md` đã được xử lý ở bước trên lên và trải nghiệm chất lượng vượt trội.

## ☁️ Deploy Lên Streamlit Community Cloud

1.  Push toàn bộ code của bạn lên một repository GitHub.
2.  Truy cập [Streamlit Community Cloud](https://streamlit.io/cloud) và tạo một ứng dụng mới từ repository của bạn.
3.  Vào phần **"Settings" -> "Secrets"** của ứng dụng và thêm API key của bạn:
    ```toml
    GROQ_API_KEY = "gsk_YourGroqApiKeyHere"
    ```
4.  Nhấn "Save" và deploy. Ứng dụng của bạn sẽ hoạt động online!
