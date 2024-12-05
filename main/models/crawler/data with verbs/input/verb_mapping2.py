import json
import os
from openai import OpenAI
import logging

# 로그 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def create_prompt(data):
    """간소화된 프롬프트 생성"""
    prompt = f"""
    8가지 카테고리로 all_keywords 내 명사들과 relations 내의 동사 관계를 분류하세요:
    1: 인과/상관성, 2: 변화/추세, 3: 시장/거래, 
    4: 정책/제도, 5: 기업 활동, 6: 금융 상품/자산, 
    7: 위험/위기, 8: 기술/혁신

    관계 데이터:
    {json.dumps(data['relations'], indent=2, ensure_ascii=False)}

    JSON 형식으로 분류 결과 반환
    결과 예시)
    {{
      "title": "마크로젠, 국내 최초 비의료기관 바이오뱅크 개설 허가",
      "relations": {{
        "3": [["신진", "브랜드", "발굴하다"]],
        "5": [["팝업", "고객", "얻다"]],
        "8": [["인터내셔날", "공간", "적용하다"]]
      }}
    }},
    """
    return prompt.strip()

def classify_relations(client, data):
    """OpenAI API 호출 및 응답 처리"""
    prompt = create_prompt(data)
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant analyzing verb-noun relationships."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        
        # 결과가 비어있지 않은 경우에만 반환
        if result:
            # 중첩된 relations 제거
            flat_relations = {}
            for key, value in result['relations'].items():
                if value:  # 비어있지 않은 경우만 추가
                    flat_relations[key] = value
            
            return {
                "title": data["title"],
                "url": data["url"],
                "content": data["content"],
                "all_keywords": data.get("all_keywords", []),
                "relations": flat_relations  # 중첩 제거된 relations
            }
        else:
            return None

    except json.JSONDecodeError as e:
        logging.error(f"JSON 파싱 오류: {e}")
        logging.error(f"응답 내용: {response.choices[0].message.content.strip()}")
        return None
    except Exception as e:
        logging.error(f"API 호출 중 오류: {e}")
        return None

def process_json_files(input_directory, output_directory):
    """JSON 파일 처리 함수"""
    client = OpenAI(api_key="")
    os.makedirs(output_directory, exist_ok=True)

    results = []  # 결과를 저장할 리스트

    for filename in os.listdir(input_directory):
        if filename.endswith('.json'):
            input_path = os.path.join(input_directory, filename)
            with open(input_path, 'r', encoding='utf-8') as f:
                input_data = json.load(f)

            for item in input_data:
                title = item.get("title")
                url = item.get("url")
                content = item.get("content", "")
                all_keywords = item.get("all_keywords", [])
                
                # 관계 분류
                relations_result = classify_relations(client, item)

                if relations_result:
                    results.append({
                        "title": title,
                        "url": url,
                        "contents": content,
                        "all_keywords": all_keywords,
                        "relations": relations_result["relations"]
                    })

            # 모든 item이 처리된 후에 로그 남기기
            logging.info(f"Processed file: {filename}")

    # 결과가 있는 경우에만 파일로 저장
    if results:
        output_filename = 'classified_results.json'
        output_path = os.path.join(output_directory, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    input_directory = '.'  # 입력 파일이 있는 디렉토리
    output_directory = '.'  # 결과 파일을 저장할 디렉토리
    process_json_files(input_directory, output_directory)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()