# preprocess_pdf.py

import os
from marker.models import load_all_models
from marker.convert import convert_single_pdf
from pathlib import Path
import argparse

def process_pdf_to_markdown(pdf_path: str, output_dir: str):
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        print(f"Lỗi: Đường dẫn không phải là một file: {pdf_path}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_filename = pdf_path.stem + ".md"
    markdown_filepath = output_dir / markdown_filename

    print("Bắt đầu tải các model của Marker (chỉ lần đầu)...")
    model_lst = load_all_models()
    print("✅ Đã tải xong model.")

    print(f"Bắt đầu xử lý file: {pdf_path.name}...")
    full_text, out_meta = convert_single_pdf(str(pdf_path), model_lst)

    with open(markdown_filepath, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"🎉 Chuyển đổi thành công! File Markdown đã được lưu tại: {markdown_filepath}")

def main():
    parser = argparse.ArgumentParser(description="Chuyển đổi một file PDF hoặc một thư mục PDF thành Markdown bằng Marker.")
    parser.add_argument("input_path", type=str, help="Đường dẫn đến file PDF hoặc thư mục chứa các file PDF.")
    parser.add_argument("output_dir", type=str, help="Thư mục để lưu các file Markdown đầu ra.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if input_path.is_file():
        process_pdf_to_markdown(str(input_path), args.output_dir)
    elif input_path.is_dir():
        print(f"Đang xử lý tất cả các file PDF trong thư mục: {input_path}")
        for pdf_file in input_path.glob("*.pdf"):
            process_pdf_to_markdown(str(pdf_file), args.output_dir)
    else:
        print(f"Lỗi: Đường dẫn không hợp lệ: {input_path}")

if __name__ == "__main__":
    main()
    # Cách chạy từ terminal:
    # python preprocess_pdf.py documents/pdfs documents/markdowns
