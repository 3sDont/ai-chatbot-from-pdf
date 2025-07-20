# src/llm_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLMModel:
    def __init__(self, model_name="vinai/PhoGPT-4B-Chat"):
        # Thêm trust_remote_code=True nếu dùng các model như microsoft/phi-2
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        self.model.eval()

    def generate_answer(self, context, question, max_new_tokens=300):
        prompt = (
            f"Sử dụng ngữ cảnh dưới đây để trả lời câu hỏi một cách chi tiết. "
            f"Nếu không có thông tin trong ngữ cảnh, hãy nói rằng 'Tôi không tìm thấy thông tin này trong tài liệu.'\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {question}\n"
            f"Trả lời:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response_text.split("Trả lời:")[-1].strip()
        return answer
