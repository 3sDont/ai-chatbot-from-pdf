# src/components/data_loader.py

class DataLoader:
    def load(self, file_path: str) -> str:
        """Đọc nội dung từ một file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            print(f"Lỗi khi đọc file {file_path}: {e}")
            return ""
