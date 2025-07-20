# src/llm_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLMModel:
    def __init__(self, model_name="microsoft/phi-2"):
        print("⏳ Đang load mô hình LLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float32)
        self.model.eval()
        print("✅ Đã load mô hình.")

    def generate_answer(self, context, question, max_tokens=300):
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
