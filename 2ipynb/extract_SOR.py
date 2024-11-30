import json
import pandas as pd
import nltk
from konlpy.tag import Komoran
from tqdm import tqdm
import re

class ArticleAnalyzer:
    def __init__(self):
        self.komoran = Komoran()
        nltk.download('punkt', quiet=True)
        
    def preprocess_text(self, text):
        """텍스트를 문장 단위로 분리하고 전처리"""
        # 특수문자 처리
        text = re.sub(r'["\'"」「◇△▲]', '', text)
        text = text.replace('○', '. ')
        text = text.replace('..', '.')
        
        # 문장 분리
        sentences = nltk.sent_tokenize(text)
        return [sent.strip() for sent in sentences if len(sent.strip()) > 10]

    def extract_components(self, sentence):
        """문장에서 주어, 목적어, 서술어 추출"""
        morphs = self.komoran.pos(sentence)
        
        subject = []
        object = []
        predicate = []
        temp = []
        
        for word, tag in morphs:
            # 체언 처리
            if tag.startswith('NN') or tag.startswith('NP'):
                temp.append(word)
            # 주격 조사 처리
            elif tag == 'JKS' and temp:
                subject.append(''.join(temp))
                temp = []
            # 목적격 조사 처리
            elif tag == 'JKO' and temp:
                object.append(''.join(temp))
                temp = []
            # 서술어 처리
            elif tag in ['VV', 'VA', 'XSV']:
                if temp:
                    predicate.append(''.join(temp) + word)
                    temp = []
                else:
                    predicate.append(word)
            # 문장 부호나 다른 조사를 만나면 초기화
            elif tag in ['SF', 'SP', 'SS']:
                if temp:
                    if predicate:
                        object.append(''.join(temp))
                    temp = []
        
        # 마지막 temp 처리
        if temp:
            if predicate:
                object.append(''.join(temp))
            temp = []
            
        # 결과 정제
        subject = ' '.join(subject)
        object = ' '.join(object)
        relation = ' '.join(predicate)
        
        # 최소한의 필터링만 적용
        if subject or object:  # 주어나 목적어 중 하나라도 있으면 반환
            return {
                'subject': subject,
                'object': object,
                'relation': relation
            }
        return None

    def process_article(self, article):
        """기사 전체 처리"""
        article_id = article['article_id']
        title = article['title']
        content = article['content']
        
        sentences = self.preprocess_text(content)
        results = []
        
        for sentence in sentences:
            try:
                components = self.extract_components(sentence)
                if components:
                    results.append({
                        'article_id': article_id,
                        'title': title,
                        'sentence': sentence,
                        'subject': components['subject'],
                        'object': components['object'],
                        'relation': components['relation']
                    })
            except Exception as e:
                print(f"Error processing sentence: {sentence}")
                print(f"Error: {str(e)}")
                
        return results

def main():
    analyzer = ArticleAnalyzer()
    
    with open('input.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    all_results = []
    for article in tqdm(data['articles'], desc="Processing articles"):
        results = analyzer.process_article(article)
        all_results.extend(results)
    
    df = pd.DataFrame(all_results)
    
    # 완전히 빈 컬럼이 있는 행 제거
    df = df[~(df['subject'].isna() & df['object'].isna() & df['relation'].isna())]
    
    df.to_csv('article_analysis_relaxed.csv', encoding='utf-8-sig', index=False)
    print(f"분석 완료: {len(df)} 문장이 처리되었습니다.")
    
    # 샘플 결과 출력
    print("\n추출된 결과 샘플:")
    print(df[['sentence', 'subject', 'object', 'relation']].head())

if __name__ == "__main__":
    main()