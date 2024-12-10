# src/file_loader.py
# 텍스트 파일을 불러와 내용을 문자열로 반환
def load_article(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
