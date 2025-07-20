# src/llm_model.py

from transformers import pipeline
import torch

class LLMModel:
    def __init__(self, model_name: str = "vinai/bartpho-word"):
        """
        Khởi tạo một pipeline Question Answering.
        """
        device = 0 if torch.cuda.is_available() else -1
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=device
        )
        # Lấy độ dài tối đa mà model có thể xử lý từ tokenizer
        self.max_length = self.qa_pipeline.tokenizer.model_max_length

    def generate_answer(self, context: str, question: str, **kwargs) -> str:
        """
        Sử dụng pipeline để tìm câu trả lời trong context.
        Xử lý trường hợp context quá dài.
        """
        if not context or not question:
            return "Ngữ cảnh hoặc câu hỏi không được để trống."

        # ====================================================================
        # SỬA LỖI TYPEERROR: CẮT BỚT CONTEXT NẾU QUÁ DÀI
        # Tránh đưa context quá dài vào model, gây lỗi
        # Chúng ta trừ đi độ dài câu hỏi và một chút buffer
        max_context_len = self.max_length - len(self.qa_pipeline.tokenizer.encode(question)) - 5 
        if len(context) > max_context_len:
            context = context[:max_context_len]
        # ====================================================================
            
        try:
            result = self.qa_pipeline(question=question, context=context)
            answer = result.get('answer', "Không tìm thấy câu trả lời phù hợp trong văn bản.")
            
            # Cải thiện định dạng câu trả lời
            if result.get('score', 0) < 0.1: # Nếu độ tin cậy thấp
                 return "Tôi không chắc chắn, nhưng có vẻ thông tin liên quan là: " + answer

            return answer.capitalize() + "."
            
        except Exception as e:
            print(f"Lỗi trong quá trình QA pipeline: {e}")
            return "Đã xảy ra lỗi khi xử lý yêu cầu của bạn."
