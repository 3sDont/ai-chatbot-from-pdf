# src/llm_model.py

from transformers import pipeline
import torch

class LLMModel:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Khởi tạo pipeline "text2text-generation" với model Flan-T5.
        Model này rất giỏi trong việc làm theo hướng dẫn và tóm tắt.
        """
        device = 0 if torch.cuda.is_available() else -1
        self.pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            device=device,
            max_length=512  # Giới hạn độ dài câu trả lời
        )

    def generate_answer(self, context: str, question: str, **kwargs) -> str:
        """
        Tạo prompt hướng dẫn cho Flan-T5 và sinh câu trả lời.
        """
        # Tạo một prompt rõ ràng, ra lệnh cho model phải làm gì.
        # Đây là cách hiệu quả nhất để làm việc với các model instruction-tuned.
        prompt = f"""
        Dựa vào ngữ cảnh dưới đây:
        ---
        {context}
        ---
        Hãy trả lời câu hỏi sau: "{question}"
        """
        
        try:
            # Sinh câu trả lời từ prompt
            results = self.pipeline(prompt)
            # Lấy phần text được sinh ra
            answer = results[0]['generated_text']
            return answer
            
        except Exception as e:
            print(f"Lỗi trong quá trình sinh văn bản: {e}")
            return "Rất tiếc, đã có lỗi xảy ra khi mô hình AI đang suy nghĩ."
