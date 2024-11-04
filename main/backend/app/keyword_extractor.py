from keybert import KeyBERT
from kiwipiepy import Kiwi
import time

# 참고용 경제 용어 사전 구축


# 키워드 추출하기
class KeywordExtractor:
    def __init__(self):
        self.model = KeyBERT(model='multi-qa-mpnet-base-cos-v1')
        self.kiwi = Kiwi()

    def extract_nouns(self, text):
        """명사만 추출하는 함수"""
        nouns = []
        result = self.kiwi.analyze(text)
        for token in result[0][0]:
            if token.tag.startswith('N'):
                nouns.append(token.form)
        return ' '.join(nouns)

    def extract_keywords(self, text):
        """키워드 추출"""
        # 1. 먼저 명사만 추출
        nouns_text = self.extract_nouns(text)
        
        start_time = time.time()
        
        # 2. 추출된 명사들에서 키워드 추출
        keywords = self.model.extract_keywords(
            nouns_text,
            keyphrase_ngram_range=(1, 1),
            stop_words=None,
            top_n=10,
            use_maxsum=True,
            use_mmr=True,
            diversity=0.8
        )
        
        # 키워드만 추출 (점수 제외)
        keywords = [kw[0] for kw in keywords]
        
        end_time = time.time()
        print(f"키워드 추출 시간: {end_time - start_time:.2f}초")
        
        return keywords