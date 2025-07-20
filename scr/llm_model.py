# src/llm_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLMModel:
    def __init__(self, model_name="microsoft/phi-2"):
        # Các model và tokenizer sẽ được cache bởi Streamlit trong app.py
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto", # Dùng auto để tương thích tốt hơn
            trust_remote_code=True # Cần cho model phi
        )
        self.model.eval()

    def generate_answer(self, context, question, max_new_tokens=300):
        prompt = (
            f"Sử dụng ngữ cảnh dưới đây để trả lời câu hỏi. Nếu không biết câu trả lời, hãy nói rằng bạn không tìm thấy thông tin trong tài liệu.\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {question}\n"
            f"Trả lời:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # Lấy phần text được sinh ra sau prompt
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Tách câu trả lời ra khỏi prompt
        answer = response_text.split("Trả lời:")[-1].strip()
        return answer
