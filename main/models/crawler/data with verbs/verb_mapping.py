import json
from openai import OpenAI

def create_prompt(data):
    prompt = f"""
    당신은 경제 및 금융 분야의 전문가입니다. 아래는 8가지 동사 카테고리입니다. 각 동사는 명사와 결합하여 적절한 카테고리에 속합니다.

    '1': '인과 혹은 상관성'
    '2': '변화 & 추세'
    '3': '시장 및 거래 관계'
    '4': '정책 & 제도'
    '5': '기업 활동'
    '6': '금융 상품 & 자산'
    '7': '위험 및 위기'
    '8': '기술 및 혁신'

    주어진 JSON 데이터에서 각 `relations` 배열의 동사와 명사 관계를 읽고, 이를 위 카테고리에 맞게 분류하세요. 결과는 기존 데이터 구조를 유지하며, `relations`를 카테고리별로 분류한 형태로 반환합니다.
    단, 매핑이 되지 않는 동사 표현에 대해선 따로 처리하지 않아도 됩니다. 

    JSON 데이터:
    {json.dumps(data, indent=2, ensure_ascii=False)}

    결과는 다음 JSON 형식으로 작성하세요:
        {
            "title": "{data['title']}",
            "url": "{data['url']}",
            "all_keywords": {json.dumps(data['all_keywords'], ensure_ascii=False)},
            "relations": {
                {
                    "1": [...],
                    "2": [...],
                    "3": [...],
                    "4": [...],
                    "5": [...],
                    "6": [...],
                    "7": [...],
                    "8": [...]
                }
            }
        }
    """
    return prompt.strip()

def classify_relations(client, data):
    prompt = create_prompt(data)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API 호출 중 오류 발생: {e}")
        return None

def save_result_to_json(result, filename='verb_mapping_result.json'):
    try:
        parsed_result = json.loads(result)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(parsed_result, f, ensure_ascii=False, indent=2)
        print(f"결과를 {filename}에 저장했습니다.")
    except json.JSONDecodeError:
        print("JSON 파싱 중 오류 발생. 결과를 저장할 수 없습니다.")
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")

def main():
    # OpenAI API 키 설정
    client = OpenAI(api_key="")

    # 입력 JSON 데이터
    input_data = {
        "title": "마크로젠, 국내 최초 비의료기관 바이오뱅크 개설 허가",
        "url": "https://www.hankyung.com/article/2024060447515",
        "all_keywords": [
            "글로벌", "연구", "프로젝트", "인공지능", "디지털", "바이오", "검진", 
            "건강", "센터", "관리청", "헬스케어", "개설", "인증", "의료", "이용",
            "올오브어스", "은행", "로젠", "뱅크", "질병", "유래", "마크", "미국", 
            "마크로젠", "인체"
        ],
        "relations": [
            ["개설하다", "마크로젠", "바이오뱅크"],
            ["확보하다", "마크로젠", "데이터·바이오뱅크"],
            ["기여하다", "마크로젠", "의료비 절감"]
        ]
    }

    # 관계 분류
    result = classify_relations(client, input_data)
    
    # 결과 저장
    if result:
        save_result_to_json(result)
        print("\n분석 결과:")
        print(result)

if __name__ == "__main__":
    main()