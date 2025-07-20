# src/pipelines/llm_models.py

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

class GroqLLM:
    def __init__(self, model_name: str = "llama3-8b-8192"):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY không được tìm thấy. Vui lòng thêm vào Streamlit Secrets.")
            
        self.chat = ChatGroq(temperature=0.1, model_name=model_name, api_key=api_key)
        
        system_prompt = (
            "Bạn là một trợ lý AI chuyên gia, có khả năng phân tích và tóm tắt tài liệu. "
            "Hãy trả lời câu hỏi của người dùng một cách chính xác, dựa hoàn toàn vào 'Ngữ cảnh' được cung cấp. "
            "Nếu thông tin không có trong ngữ cảnh, hãy trả lời là 'Tôi không tìm thấy thông tin này trong tài liệu.' "
            "Trình bày câu trả lời một cách rõ ràng, mạch lạc bằng tiếng Việt."
        )
        human_prompt = "Ngữ cảnh: {context}\n\nCâu hỏi: {question}"
        
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
        output_parser = StrOutputParser()
        self.chain = prompt | self.chat | output_parser

    def generate(self, context: str, question: str) -> str:
        try:
            return self.chain.invoke({"context": context, "question": question})
        except Exception as e:
            return f"Lỗi khi gọi Groq API: {e}"
