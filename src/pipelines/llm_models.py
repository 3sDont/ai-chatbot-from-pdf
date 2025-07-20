# src/pipelines/llm_models.py

from transformers import pipeline
import torch

class FlanT5:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        device = 0 if torch.cuda.is_available() else -1
        self.pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            device=device,
            max_length=512
        )

    def generate(self, context: str, question: str) -> str:
        prompt = f"""
        Dựa vào ngữ cảnh dưới đây:
        ---
        {context}
        ---
        Hãy trả lời câu hỏi sau: "{question}"
        """
        try:
            results = self.pipeline(prompt)
            return results[0]['generated_text']
        except Exception as e:
            print(f"Lỗi khi sinh văn bản: {e}")
            return "Rất tiếc, đã có lỗi xảy ra khi mô hình AI đang suy nghĩ."
