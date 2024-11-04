import pandas as pd
import json

# 파일 경로 설정
csv_file_path = '1 한국산업은행_금융 관련 용어.csv'
json_file_path = '700_dict.json'
updated_json_file_path = 'updated1_700_dict.json'

# CSV 파일 읽기
csv_df = pd.read_csv(csv_file_path, encoding='euc-kr')

# JSON 파일 로드
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    json_data = json.load(json_file)

# 기존 JSON에서 'word' 값 추출
json_words = {entry.get("word", "") for entry in json_data if "word" in entry}

# CSV 파일에서 'word' 값 추출
csv_words = set(csv_df.iloc[:, 0])

# CSV에만 있는 단어 추출
new_words = csv_words - json_words

# 새로운 단어들을 JSON 형식으로 추가
for word in new_words:
    json_data.append({
        "word": word,
        "relation_word": []
    })

# 업데이트된 JSON 파일 저장
with open(updated_json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=4)

print(f"Updated JSON file saved to {updated_json_file_path}")
