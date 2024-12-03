import json
import random
import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key='')

class ArticleProcessor:
    def __init__(self, model: str = "gpt-4o"):
        """Initialize the ArticleProcessor with specified GPT model."""
        self.model = model

    async def extract_verbs_from_text(self, text: str) -> List[str]:
        """
        Extract verbs from text using OpenAI GPT API.
        
        Args:
            text (str): Input text to process
            
        Returns:
            List[str]: List of extracted verbs
        """
        example = {
            "title": "시중은행에 치이고 인뱅에 밀린 지방은행",
            "content": "지역 경제의 버팀목 역할을 해온 지방은행이 설 자리를 잃고 있다. 자금력이 우수한 시중은행이 지방 영업을 확대하는 가운데 개인 고객마저 디지털 금융을 앞세운 인터넷전문은행에 빼앗기면서다...",
            "all_keywords": ["지역 경제", "지방은행", "자금력", "시중은행", "영업 확대", "개인 고객", "디지털 금융", "인터넷전문은행"],
            "verbs": ["영향을 미치다", "확장하다", "감소하다"] 
        }
        
        prompt = f"""
        넌 전세계 경제, 금융 이슈들과 그 핵심 키워드들 간의 관계에 대해 잘 알고 있는 경제 분야 전문가야.
        뉴스 전문을 보고 핵심 키워드를 문장 내에서 파악한 뒤, 그 키워드들 간의 관계를 동사 표현을 기반으로 설명해야 해.
        
        규칙:
        1. 뉴스 내용에서 핵심 키워드들 사이의 관계를 나타내는 동사만 추출
        2. 모든 동사는 반드시 동사원형으로 변환
        3. 동사 표현만 추출하고 다른 설명은 제외
        4. 중복되는 동사도 모두 포함
        
        예시 텍스트:
        {example['content']}

        예시 핵심 키워드:
        {example['all_keywords']}
        
        예시 추출 결과:
        {example['verbs']}
        
        이제 아래 텍스트를 분석해서 동사를 추출해줘:
        {text}
        """

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            result = response.choices[0].message.content.strip()
            # 리스트 형태로 반환된 결과를 처리
            if result.startswith('[') and result.endswith(']'):
                result = result[1:-1]  # 대괄호 제거
            return [verb.strip().strip('"\'') for verb in result.split(',') if verb.strip()]
        except Exception as e:
            print(f"Error extracting verbs: {e}")
            return []

    @staticmethod
    def read_random_articles(file_path: str) -> List[Dict]:
        """
        Read random articles from JSON file.
        
        Args:
            file_path (str): Path to JSON file
            sample_size (int): Number of articles to sample
            
        Returns:
            List[Dict]: List of sampled articles
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                articles = data if isinstance(data, list) else []
                return articles
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError:
            print(f"Invalid JSON format in file: {file_path}")
            return []
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []

    @staticmethod
    def save_processed_articles(file_path: str, articles: List[Dict]) -> bool:
        """
        Save processed articles to JSON file.
        
        Args:
            file_path (str): Output file path
            articles (List[Dict]): Processed articles to save
            
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(articles, file, ensure_ascii=False, indent=4)
            print(f"Processed articles saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving file {file_path}: {e}")
            return False

async def main():
    # Configuration
    json_file_path = "key_extract_final_1.json"
    output_file_path = "key_extract_final_1_verbs.json"
    
    # Initialize processor
    processor = ArticleProcessor()
    
    # Process articles
    random_articles = processor.read_random_articles(json_file_path)
    
    if random_articles:
        processed_articles = []
        for i, article in enumerate(random_articles, start=1):
            content = article.get("content", "")
            if content:
                print(f"Processing article {i}")
                verbs = await processor.extract_verbs_from_text(content)
                article["verbs"] = verbs
                processed_articles.append(article)
        
        processor.save_processed_articles(output_file_path, processed_articles)
    else:
        print("No articles could be read from the input file.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())