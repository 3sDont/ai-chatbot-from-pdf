# src/llm_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLMModel:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto", # Tự động chọn dtype phù hợp
            trust_remote_code=True
        )
        self.model.eval()

    def generate_answer(self, context: str, question: str, max_new_tokens: int = 512) -> str:
        """Tạo câu trả lời từ LLM dựa trên context và câu hỏi."""
        prompt = (
            "Dựa vào ngữ cảnh được cung cấp dưới đây, hãy trả lời câu hỏi của người dùng một cách đầy đủ và chi tiết. "
            "Chỉ sử dụng thông tin từ ngữ cảnh. Nếu câu trả lời không có trong ngữ cảnh, hãy nói rõ 'Tôi không tìm thấy thông tin này trong tài liệu.'\n\n"
            f"--- NGỮ CẢNH ---\n{context}\n\n"
            f"--- CÂU HỎI ---\n{question}\n\n"
            "--- TRẢ LỜI ---"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Tách câu trả lời ra khỏi prompt một cách an toàn
        answer_parts = response_text.split("--- TRẢ LỜI ---")
        return answer_parts[-1].strip() if len(answer_parts) > 1 else "Không thể tạo câu trả lời."
