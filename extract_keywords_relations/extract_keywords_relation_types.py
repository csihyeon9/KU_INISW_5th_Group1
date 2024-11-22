import pandas as pd
import openai
import json
from typing import List, Dict
import time

class ArticleAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key
        
        self.relation_types = [
            "경영 전략", "조직 관리", "운영 관리", "마케팅", "윤리 경영", 
            "거시 경제", "미시 경제", "국제 경제", "정책 경제학", "산업 경제학"
            "공공 정책", "행정 관리", "규제와 법률", "국제 관계", "사회적 가치",
            "기술 혁신", "환경 과학", "산업 기술", "데이터 과학", "학문적 발전",
            "금융 시장", "금융 상품", "금융 기술", "자산 관리", "글로벌 금융",
            "사회 문제", "문화와 소비", "교육과 노동", "건강과 복지", "윤리와 가치"
            ]
        
    def create_prompt(self, article_content: str) -> str:
        example_json = '''
    {
        "categories": [
            {
                "relation_type": "분류1",
                "keywords": ["키워드1", "키워드2", "키워드3"]
            },
            {
                "relation_type": "분류2",
                "keywords": ["키워드4", "키워드5", "키워드6"]
            }
        ]
    }'''

        return f"""다음 기사를 분석하여 각 relation_type별로 관련된 keywords를 추출해주세요.

    기사 내용:
    {article_content}

    Here are the rules you must strictly follow:

relation_type should only be selected from the following list: {', '.join(self.relation_types)}

keywords must only include professional economic/financial terms:

Do not include company names and personal names.
Exclude general words. Include mentioned terms that fit in the relation_types catergory only.
Only include strictly economic/financial terminology.
Again, the keywords must only contain legitimate economic/financial terms. Company names, personal names, and unrelated terms must not be included. Please be sure to carefully consider and include only authentic economic and financial terms. This is a very important project for our team, so please make sure to follow these instructions carefully. Please.

Extract distinct keywords for each relation_type.

    다음 JSON 형식으로 출력해주세요:
    {example_json}"""

    def analyze_article(self, content: str) -> Dict:
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # GPT-4 대신 GPT-3.5-turbo 사용
                messages=[
                    {"role": "system", "content": "당신은 경제 금융 전문가입니다. 기사를 분석하여 각 분야별로 언급된 경제 키워드만을 추출합니다."},
                    {"role": "user", "content": self.create_prompt(content)}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"Error analyzing article: {e}")
            return {"categories": []}

    # def process_articles(self, input_path: str, output_path: str):
    #     # 엑셀 파일 읽기
    #     df = pd.read_excel(input_path)
        
    #     # date 컬럼이 문자열인 경우 처리
    #     results = []
        
    #     for idx, row in df.iterrows():
    #         print(f"Processing article {idx + 1}/{len(df)}")
            
    #         analysis_result = self.analyze_article(row['content'])
            
    #         for category in analysis_result['categories']:
    #             article_result = {
    #                 "url": row['link'],
    #                 "title": row['title'],
    #                 "journalist": row['journalist'],
    #                 "date": row['date'],  # 문자열 그대로 사용
    #                 "relation_type": category['relation_type'],
    #                 "keywords": category['keywords']
    #             }
    #             results.append(article_result)
            
    #         time.sleep(3)
            
    #         if (idx + 1) % 10 == 0:
    #             print("Taking a 10-second break to avoid rate limits...")
    #             time.sleep(10)

    def process_articles(self, input_path: str, output_path: str):
        # JSON 파일 읽기
        with open(input_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        # 결과 저장 리스트
        results = []
        
        for idx, article in enumerate(articles):
            print(f"Processing article {idx + 1}/{len(articles)}")
            
            try:
                analysis_result = self.analyze_article(article['content'])
            except Exception as e:
                # 오류가 발생한 기사의 URL을 출력하여 어떤 기사에서 오류가 발생했는지 알 수 있도록 함 히히
                print(f"Error processing article {article['url']}: {e}")
                continue  # 오류가 발생한 기사는 건너뛰고, 다음 기사로 넘어감
            
            for category in analysis_result['categories']:
                article_result = {
                    "url": article['url'],
                    "title": article['title'],
                    "journalist": article.get('journalist', 'Unknown'),
                    "date": article['time'], 
                    "relation_type": category['relation_type'],
                    "keywords": category['keywords']
                }
                results.append(article_result)
            
            time.sleep(3)  # API 호출 간의 대기 시간을 유지 (필요시 조정 가능)
        
        # 결과를 JSON 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"Analysis completed. Results saved to {output_path}")


def main():
    # API 키 설정
    API_KEY = "ㅎㅣ히히히히히히히히"
    
    # 분석기 초기화
    analyzer = ArticleAnalyzer(API_KEY)
    
    # 파일 경로 설정
    input_file = "hankyung_news_3000_extracted_articles.json"
    output_file = "analyzed_articles.json"
    
    # 처리 실행
    analyzer.process_articles(input_file, output_file)

if __name__ == "__main__":
    main()
