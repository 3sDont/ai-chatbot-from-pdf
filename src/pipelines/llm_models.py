# src/pipelines/llm_models.py

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

class GroqLLM:
    def __init__(self, model_name: str = "llama3-8b-8192"):
        """
        Khởi tạo mô hình LLM sử dụng Groq API.
        Sử dụng LangChain để dễ dàng tích hợp.
        """
        try:
            # Tự động đọc API key từ Streamlit Secrets
            self.chat = ChatGroq(
                temperature=0,
                model_name=model_name,
                api_key=os.environ.get("GROQ_API_KEY") 
            )
        except Exception as e:
            raise ValueError(f"Không thể khởi tạo Groq. Hãy chắc chắn bạn đã thêm GROQ_API_KEY vào Streamlit Secrets. Lỗi: {e}")

        # Định nghĩa một chuỗi xử lý (chain) với LangChain
        # 1. Prompt Template: Định nghĩa cấu trúc prompt
        system_prompt = (
            "Bạn là một trợ lý AI hữu ích. Hãy trả lời câu hỏi của người dùng chỉ dựa vào ngữ cảnh được cung cấp. "
            "Trả lời một cách ngắn gọn, súc tích và chính xác. Nếu không tìm thấy thông tin, hãy nói rõ 'Tôi không tìm thấy thông tin này trong tài liệu.'"
        )
        human_prompt = "Ngữ cảnh: {context}\n\nCâu hỏi: {question}"
        
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
        
        # 2. Output Parser: Chuyển đổi output của model thành chuỗi string
        output_parser = StrOutputParser()

        # 3. Kết hợp lại thành một chain
        self.chain = prompt | self.chat | output_parser

    def generate(self, context: str, question: str) -> str:
        """
        Sử dụng chain đã định nghĩa để tạo câu trả lời.
        """
        try:
            return self.chain.invoke({"context": context, "question": question})
        except Exception as e:
            print(f"Lỗi khi gọi Groq API: {e}")
            return "Rất tiếc, đã xảy ra lỗi khi kết nối đến mô hình AI."
