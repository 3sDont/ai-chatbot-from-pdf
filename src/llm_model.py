# src/llm_model.py

from transformers import pipeline
import torch

class LLMModel:
    def __init__(self, model_name: str = "vinai/bartpho-word"):
        """
        Khởi tạo một pipeline Question Answering.
        Đây là cách hiệu quả để dùng các model như BART, RoBERTa cho tác vụ RAG.
        """
        # Xác định thiết bị (sử dụng GPU nếu có)
        device = 0 if torch.cuda.is_available() else -1
        
        # Tạo pipeline với model và tokenizer được chỉ định
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=device
        )

    def generate_answer(self, context: str, question: str, **kwargs) -> str:
        """
        Sử dụng pipeline để tìm câu trả lời trong context.
        """
        if not context or not question:
            return "Ngữ cảnh hoặc câu hỏi không được để trống."

        # pipeline sẽ trả về một dictionary
        result = self.qa_pipeline(question=question, context=context)
        
        # Lấy câu trả lời có điểm số cao nhất
        answer = result.get('answer', "Không tìm thấy câu trả lời phù hợp trong văn bản.")
        
        # Các model QA đôi khi trả về câu trả lời rất ngắn. 
        # Chúng ta có thể làm nó tự nhiên hơn một chút.
        if result['score'] < 0.1: # Nếu độ tin cậy thấp
             return "Tôi không chắc chắn, nhưng thông tin liên quan có thể là: " + answer

        return answer.capitalize() + "."
