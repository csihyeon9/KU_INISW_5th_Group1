# extractor.py

from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel
import torch
from keybert import KeyBERT
from bs4 import BeautifulSoup
from kiwipiepy import Kiwi

# 경제 및 금융 용어 리스트 (키워드 추출 시 기준이 되는 용어들)
ECONOMIC_TERMS = set([
    '금리', '주식', '부동산', '채권', '인플레이션', '환율', '경제', '투자', '자산',
    '소득', '지출', '대출', '예금', '보험', '파산', '회계', '시장', '금융', 
    '소비자물가', '경제성장', 'GDP', '통화', '금융정책', '재정정책', '기업', '부채', '증권', 
    '펀드', 'ETF', '지수', '선물', '옵션', '파생상품', '헤지펀드', '사모펀드',
    '공모펀드', '연금', '변액보험', '적립식펀드', '채무불이행', '신용카드', '체크카드',
    '단기금융상품', '리스크관리상품', '외환파생상품', '주식파생상품', '신탁', '파킹계좌',
    '소비자물가지수', '생산자물가지수', '고용지표', '실업률', '고용률', '경제활동참가율',
    '무역수지', '경상수지', '재정수지', '재정적자', '무역적자', '외환보유고',
    '경기지수', '산업생산지수', '경제동향지수', '수입물가지수', '주가지수',
    '글로벌화', '무역장벽', '자유무역협정', 'WTO', 'IMF', '세계은행', 'FTA',
    '수출', '수입', '무역수지', '환율전쟁', '자본유출', '자본유입',
    '보호무역주의', '경상수지적자', '경상수지흑자', '경쟁적평가절하',
    '중앙은행', '한국은행', '미연준', '유럽중앙은행', '금융감독원', '예금보험공사',
    '금융위원회', '금융소비자보호법', '바젤협약', '금융규제', '금융자율화',
    '신탁회사', '자산운용사', '증권거래소', '자본시장법', '대출금리상한제', '금리자유화',
    '비트코인', '블록체인', '디지털화폐', '중앙은행디지털화폐', '토큰', '이더리움',
    '스테이블코인', '암호화폐', '핀테크', 'P2P대출', '크라우드펀딩',
    '증권형토큰공개', '인공지능투자', '로보어드바이저', '부채비율', '자산유동화',
    '경기침체', '경기과열', '경기회복', '경기불황', '디플레이션', '스태그플레이션',
    '경기확장', '경제순환', '자본축적', '경제순환이론', '경제분석', '경기지표',
    '소비심리지수', '제조업지수', '설비투자', '경기부양책', '경기조정', '경제위기',
    '기업가치', 'IPO', 'M&A', '기업공개', '상장', '비상장', '벤처캐피탈',
    '스타트업', '중소기업', '대기업', '기업분할', '기업합병', '지주회사', '자회사',
    '지배구조', '주주총회', '경영권', '경영분석', '기업자산', '가치평가', '조세회피',
    '포트폴리오', '리스크관리', '자산배분', '가치투자', '성장투자', '기술적분석',
    '기초분석', '배당', '자본이득', '단기투자', '장기투자', '공매도', '헤지',
    '삼성전자', '투자', 'KB증권', '리서치', '주식차익거래', '주가예측', '모멘텀투자',
    '분산투자', '테마주', '배당수익률', '액티브투자', '패시브투자',
    '전세', '월세', '임대차', '담보대출', 'LTV', 'DTI', '주택담보대출',
    '분양', '재건축', '재개발', '상가', '오피스텔', '주택임대사업자',
    '임대수익률', '주택연금', '상업용부동산', '토지개발', '건물관리', '지분투자',
    '개인파산', '신용대출', '학자금대출', '자동차대출', '할부', '리스', '개인연금',
    '국민연금', '퇴직연금', '개인퇴직연금', '소득공제', '세액공제', '상속세', '증여세',
    '신용평가', '파산보호', '소액대출', '소비자신용', '세금환급', '대출상환유예'
])

class Ext_Key:
    """
    Ext_Key 클래스는 주어진 HTML 파일에서 경제 및 금융 관련 용어를 추출하는 역할을 수행.
    초기화 시에 Transformer 모델과 형태소 분석기, 경제 용어 임베딩을 준비함.
    """
    
    def __init__(self, filepath):
        """
        Ext_Key 클래스 초기화
        - Transformer 모델과 형태소 분석기 및 경제 용어 사전을 준비
        
        :param filepath: 분석할 HTML 파일의 경로
        """
        self.filepath = filepath
        # Transformer 기반의 토크나이저와 모델 초기화
        self.tokenizer = AutoTokenizer.from_pretrained("upskyy/kf-deberta-multitask") 
        self.model = AutoModel.from_pretrained("upskyy/kf-deberta-multitask")
        # KeyBERT 모델 초기화
        self.keyBERT_model = KeyBERT(model='distiluse-base-multilingual-cased-v1')
        # Kiwi 형태소 분석기 초기화
        self.kiwi = Kiwi()
        # 경제 용어 사전과 해당 용어들의 임베딩을 준비
        self.economic_terms = list(ECONOMIC_TERMS)
        self.economic_embeddings = self.create_embeddings(self.economic_terms)

    def load_data(self):
        """
        HTML 파일에서 텍스트 데이터를 추출하는 함수.
        HTML 파일을 읽고 제목과 본문 텍스트를 추출하여 하나의 문자열로 반환함.
        
        :return: HTML 파일의 제목과 본문 텍스트
        """
        with open(self.filepath, 'r', encoding='utf-8') as file:
            html_content = file.read()
        dom = BeautifulSoup(html_content, 'html.parser')
        title = dom.title.string if dom.title else ""
        dic_area = dom.find(id="dic_area")
        dic_area_text = dic_area.get_text(separator=' ', strip=True) if dic_area else ""
        text = f"{title}\n{dic_area_text}"
        return text

    def extract_nouns_Kiwi(self, text):
        """
        Kiwi 형태소 분석기를 사용해 텍스트에서 명사만 추출하는 함수.
        
        :param text: 분석할 텍스트
        :return: 텍스트에서 추출된 명사의 리스트
        """
        nouns = []
        result = self.kiwi.analyze(text)
        for token in result[0][0]:
            if token.tag.startswith('N'):
                nouns.append(token.form)
        return nouns

    def create_embeddings(self, terms):
        """
        주어진 단어 리스트의 임베딩을 생성하는 함수.
        Transformer 모델을 사용해 임베딩을 생성하고, 평균을 통해 단어 수준의 벡터를 생성함.
        
        :param terms: 임베딩을 생성할 단어 리스트
        :return: 생성된 임베딩 텐서
        """
        inputs = self.tokenizer(terms, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    def ext_finan_terms(self):
        """
        경제 및 금융 관련 용어를 추출하는 메인 함수.
        기사 내에서 추출한 명사들과 경제 용어 사전 간의 코사인 유사도를 계산하고, 상위 20개의 용어를 반환.
        
        :return: 상위 20개의 경제 및 금융 관련 용어 리스트
        """
        # HTML에서 텍스트 추출 및 명사 추출
        text = self.load_data()
        nouns = self.extract_nouns_Kiwi(text)
        noun_embeddings = self.create_embeddings(nouns)

        unique_terms = set()
        similar_terms = []

        # 명사와 경제 용어 사전 임베딩 간 코사인 유사도를 계산
        for i, noun_embedding in enumerate(noun_embeddings):
            cosine_scores = util.cos_sim(noun_embedding.unsqueeze(0), self.economic_embeddings)[0]
            max_similarity = max(cosine_scores).item()

            # 유사도가 0.3 이상이고 중복되지 않은 명사만 추가
            if max_similarity > 0.3 and nouns[i] not in unique_terms:
                similar_terms.append((nouns[i], max_similarity))
                unique_terms.add(nouns[i])

        # 상위 20개의 키워드 반환
        similar_terms = sorted(similar_terms, key=lambda x: x[1], reverse=True)[:20]
        return [term[0] for term in similar_terms]
