# 기사 내 키워드를 추출하는 KeyBert 함수
from keybert import KeyBERT
from kiwipiepy import Kiwi

# html 파일의 기사로부터 parsing 하기 위한 패키지
from bs4 import BeautifulSoup
import time

class Ext_Key:
    def __init__(self, filepath):
        # 새로운 html 파일 경로 초기화
        self.filepath = filepath
        # KeyBERT 모델 초기화 -> model = 'skt/kobert-base-v1', 'all-MIniLM-L6-v2'
        self.model = KeyBERT(model='multi-qa-mpnet-base-cos-v1')
        # Kiwi 형태소 분석기 초기화 
        self.kiwi = Kiwi()

    def load_data(self):
        """ html 파일을 text로 읽어오는 함수 """
        # 새로 들어오는 html 파일을 실시간으로 읽어들여서 keyword_ext() 함수에서 text 단에 넘겨주어야 함. 
        with open(self.filepath, 'r', encoding='utf-8') as file:
            html_content = file.read()

        dom = BeautifulSoup(html_content, 'html.parser')
        
        # 제목 가져오기
        title = dom.title.string if dom.title else ""
        
        # id="dic_area"인 부분의 텍스트 가져오기
        dic_area = dom.find(id="dic_area")
        dic_area_text = dic_area.get_text(separator=' ', strip=True) if dic_area else ""
        
        # 제목과 dic_area 부분을 결합한 텍스트 반환
        text = f"{title}\n{dic_area_text}"
        return text

    def extract_nouns_Kiwi(self):
        """입력받은 문서의 명사만 추출하는 함수"""

        text = self.load_data()
        nouns = []

        result = self.kiwi.analyze(text)
        for token in result[0][0]:
            if token.tag.startswith('N'):  # 모든 종류의 명사 추출
                nouns.append(token.form)
        return ' '.join(nouns)

    def keyword_ext(self):
        
        # 1. 먼저 명사만 추출
        nouns_text = self.extract_nouns_Kiwi()
        
        start_time = time.time()
        
        # 2. 추출된 명사들에서 키워드 추출
        keywords = self.model.extract_keywords(
            nouns_text,  # 명사로 이뤄진 텍스트 사용
            keyphrase_ngram_range=(1, 2),
            stop_words=None,
            top_n=10,
            use_maxsum=True,
            use_mmr=True,
            diversity=0.8
        )
        
        # 키워드만 추출 (점수 제외)
        keywords = [kw[0] for kw in keywords]
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"키워드 추출 시간: {elapsed_time:.2f}초")
        
        return keywords
