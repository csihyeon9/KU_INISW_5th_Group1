import json
import asyncio
import logging
from typing import Dict, List
from openai import AsyncOpenAI
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VerbClassifier:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.categories = {
            'i': '인과 or 상관성',
            'ii': '변화 & 추세',
            'iii': '시장 및 거래 관계',
            'iv': '정책 & 제도',
            'v': '기업 활동',
            'vi': '금융 상품 & 자산',
            'vii': '위험 및 위기',
            'viii': '기술 및 혁신'
        }

    async def classify_verbs(self, verbs: List[str]) -> Dict[str, List[str]]:
        """동사 분류 함수"""
        prompt = f"""
        아래 경제 동사들을 각 카테고리별로 분류해주세요. 각 동사는 여러 카테고리에 포함될 수 있으며, 
        반드시 하나 이상의 카테고리에 속해야 합니다.

        [카테고리]
        i. 인과 or 상관성: 영향/원인/결과 관계, 상호작용 등
        ii. 변화 & 추세: 증가/감소, 상승/하락, 안정/변동 등
        iii. 시장 및 거래: 매매/유통, 공급/수요, 교환 등
        iv. 정책 & 제도: 규제/지원, 제재/완화, 시행/적용 등
        v. 기업 활동: 경쟁/협력, 인수합병, 투자/분리 등
        vi. 금융 상품 & 자산: 발행/상환, 담보/평가, 보유 등
        vii. 위험 및 위기: 영향/타격, 파산/회복, 증감 등
        viii. 기술 및 혁신: 자동화/디지털화, 분석/보호 등

        [분류할 동사 목록]
        {', '.join(verbs)}

        반드시 아래와 같은 JSON 형식으로만 응답해주세요:
        {{"A": ["동사1", "동사2"], "B": ["동사3", "동사4"], ...}}

        다른 설명이나 부가적인 텍스트는 포함하지 말아주세요.
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "경제 뉴스의 동사를 정확하게 분류하는 전문가입니다. JSON 형식으로만 응답합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            result_text = response.choices[0].message.content.strip()
            try:
                result = json.loads(result_text)
                # 누락된 카테고리 처리
                for cat in self.categories.keys():
                    if cat not in result:
                        result[cat] = []
                return result
            except json.JSONDecodeError:
                logger.error(f"JSON 파싱 오류: {result_text}")
                return {cat: [] for cat in self.categories.keys()}

        except Exception as e:
            logger.error(f"분류 중 오류 발생: {str(e)}")
            return {cat: [] for cat in self.categories.keys()}

    async def process_articles(self, input_file: str, output_file: str):
        """기사 처리 및 동사 분류 함수"""
        try:
            logger.info(f"입력 파일 처리 중: {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)

            # 결과 데이터 구조 설정
            processed_data = {
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "categories": self.categories,
                    "total_articles": len(articles),
                    "version": "1.0"
                },
                "articles": []
            }

            # 동사 분류 작업
            for i, article in enumerate(articles, 1):
                if "verbs" in article and article["verbs"]:
                    logger.info(f"Article {i}/{len(articles)} 처리 중...")
                    categorized = await self.classify_verbs(article["verbs"])
                    article["categorized_verbs"] = categorized
                    processed_data["articles"].append(article)

            # 결과 저장
            logger.info(f"결과 파일 저장 중: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)

            logger.info("모든 처리가 완료되었습니다!")
            
        except Exception as e:
            logger.error(f"파일 처리 중 오류 발생: {str(e)}")
            raise

async def main():
    config = {
        "api_key": "",  # OpenAI API 키를 입력하세요
        "input_file": "processed_articles14.json",  # 입력 파일 경로
        "output_file": "categorized_articles.json",  # 출력 파일 경로
        "model": "gpt-3.5-turbo"  # 사용할 모델
    }

    try:
        classifier = VerbClassifier(api_key=config["api_key"], model=config["model"])
        await classifier.process_articles(config["input_file"], config["output_file"])
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {str(e)}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())