import json

# 기존 JSON 데이터 파일 경로
input_file = "data/news_raw_data_updated.json"
# 변환된 JSON 데이터 파일 경로
output_file = "data/news_data.json"

def transform_data(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    transformed_data = []
    for article in data:
        # 새로운 구조에 맞게 데이터 변환
        transformed_article = {
            "title": article.get("title", ""),
            "content": article.get("content", ""),
            "date": article.get("time", ""),  # "time" -> "date"
            "all_keywords": article.get("all_keywords", [])
        }
        transformed_data.append(transformed_article)

    # 변환된 데이터를 새로운 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(transformed_data, outfile, ensure_ascii=False, indent=4)
    print(f"Data transformed and saved to {output_path}")

# 실행
transform_data(input_file, output_file)
