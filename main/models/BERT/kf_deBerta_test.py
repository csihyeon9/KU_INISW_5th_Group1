from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel
import torch
from keybert import KeyBERT
from bs4 import BeautifulSoup
from kiwipiepy import Kiwi
import time

## 추가 태스크 : 경제, 금융 용어 사전 늘리기 

ECONOMIC_TERMS = set([
    # 일반 경제 용어
    '금리', '주식', '부동산', '채권', '인플레이션', '환율', '경제', '투자', '자산',
    '소득', '지출', '대출', '예금', '보험', '파산', '회계', '시장', '금융', 
    '소비자물가', '경제성장', 'GDP', '통화', '금융정책', '재정정책', '기업', '부채', '증권', 

    # 금융 상품 및 서비스
    '펀드', 'ETF', '지수', '선물', '옵션', '파생상품', '헤지펀드', '사모펀드',
    '공모펀드', '연금', '변액보험', '적립식펀드', '채무불이행', '신용카드', '체크카드',
    '단기금융상품', '리스크관리상품', '외환파생상품', '주식파생상품', '신탁', '파킹계좌',

    # 경제 지표 및 통계
    '소비자물가지수', '생산자물가지수', '고용지표', '실업률', '고용률', '경제활동참가율',
    '무역수지', '경상수지', '재정수지', '재정적자', '무역적자', '외환보유고',
    '경기지수', '산업생산지수', '경제동향지수', '수입물가지수', '주가지수',

    # 국제 경제 및 무역
    '글로벌화', '무역장벽', '자유무역협정', 'WTO', 'IMF', '세계은행', 'FTA',
    '수출', '수입', '무역수지', '환율전쟁', '자본유출', '자본유입',
    '보호무역주의', '경상수지적자', '경상수지흑자', '경쟁적평가절하',

    # 금융 기관 및 정책
    '중앙은행', '한국은행', '미연준', '유럽중앙은행', '금융감독원', '예금보험공사',
    '금융위원회', '금융소비자보호법', '바젤협약', '금융규제', '금융자율화',
    '신탁회사', '자산운용사', '증권거래소', '자본시장법', '대출금리상한제', '금리자유화',

    # 기타 경제 개념
    '비트코인', '블록체인', '디지털화폐', '중앙은행디지털화폐', '토큰', '이더리움',
    '스테이블코인', '암호화폐', '핀테크', 'P2P대출', '크라우드펀딩',
    '증권형토큰공개', '인공지능투자', '로보어드바이저', '부채비율', '자산유동화',

    # 경기 관련 용어
    '경기침체', '경기과열', '경기회복', '경기불황', '디플레이션', '스태그플레이션',
    '경기확장', '경제순환', '자본축적', '경제순환이론', '경제분석', '경기지표',
    '소비심리지수', '제조업지수', '설비투자', '경기부양책', '경기조정', '경제위기',

    # 기업 및 경영 용어
    '기업가치', 'IPO', 'M&A', '기업공개', '상장', '비상장', '벤처캐피탈',
    '스타트업', '중소기업', '대기업', '기업분할', '기업합병', '지주회사', '자회사',
    '지배구조', '주주총회', '경영권', '경영분석', '기업자산', '가치평가', '조세회피',

    # 투자 전략 및 분석
    '포트폴리오', '리스크관리', '자산배분', '가치투자', '성장투자', '기술적분석',
    '기초분석', '배당', '자본이득', '단기투자', '장기투자', '공매도', '헤지',
    '삼성전자', '투자', 'KB증권', '리서치', '주식차익거래', '주가예측', '모멘텀투자',
    '분산투자', '테마주', '배당수익률', '액티브투자', '패시브투자',

    # 부동산 관련 용어
    '전세', '월세', '임대차', '담보대출', 'LTV', 'DTI', '주택담보대출',
    '분양', '재건축', '재개발', '상가', '오피스텔', '주택임대사업자',
    '임대수익률', '주택연금', '상업용부동산', '토지개발', '건물관리', '지분투자',

    # 소비자 금융
    '개인파산', '신용대출', '학자금대출', '자동차대출', '할부', '리스', '개인연금',
    '국민연금', '퇴직연금', '개인퇴직연금', '소득공제', '세액공제', '상속세', '증여세',
    '신용평가', '파산보호', '소액대출', '소비자신용', '세금환급', '대출상환유예'
])


## 추가 태스크 : 1) keybert 내 기타 sbert 모델 돌려보기, 2) 경제, 금융 용어 사전 내 키워드 - 입력 키워드 간 거리 계산법 찾아보기
## 3) 다른 형태소 분석기 찾아보기, 4) keyBert 외에 다른 키워드 추출 모델 찾아보기 
class Ext_Key:
    def __init__(self, filepath):
        self.filepath = filepath
        self.tokenizer = AutoTokenizer.from_pretrained("upskyy/kf-deberta-multitask") # tokenizer 모델 
        self.model = AutoModel.from_pretrained("upskyy/kf-deberta-multitask")  # kf-deberta 기반 모델
        # self.keyBERT_model = KeyBERT(model='multi-qa-mpnet-base-cos-v1')  # KeyBert 모델 => model = 'skt/kobert-base-v1', 'all-MIniLM-L6-v2'
        self.keyBERT_model = KeyBERT(model='distiluse-base-multilingual-cased-v1')
        self.kiwi = Kiwi() # 형태소 분석기 
        self.economic_terms = list(ECONOMIC_TERMS)  # 경제, 금융 용어 사전
        self.economic_embeddings = self.create_embeddings(self.economic_terms)

    def load_data(self):
        """HTML 파일을 읽어 텍스트로 변환"""
        with open(self.filepath, 'r', encoding='utf-8') as file:
            html_content = file.read()
        dom = BeautifulSoup(html_content, 'html.parser')
        title = dom.title.string if dom.title else ""
        dic_area = dom.find(id="dic_area")
        dic_area_text = dic_area.get_text(separator=' ', strip=True) if dic_area else ""
        text = f"{title}\n{dic_area_text}"
        return text

    def extract_nouns_Kiwi(self, text):
        """Kiwi 형태소 분석기를 사용해 명사만 추출"""
        ## 추가 태스크1 : 다른 형태소 분석기 사용해보기 ex) Konlpy, mecab
        nouns = []
        result = self.kiwi.analyze(text)
        for token in result[0][0]:
            if token.tag.startswith('N'):
                nouns.append(token.form)
        return nouns

    def create_embeddings(self, terms):
        """주어진 단어 리스트의 임베딩 생성"""
        inputs = self.tokenizer(terms, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    def euclidean_distance(self, v1, v2):
        """유클리디안 거리 계산"""
        return torch.dist(v1, v2, p=2)

    def manhattan_distance(self, v1, v2):
        """맨해튼 거리 계산"""
        return torch.dist(v1, v2, p=1)

    def dot_product(self, v1, v2):
        """점곱 계산"""
        return torch.dot(v1, v2)
    
    def ext_finan_terms(self):
        """ 기사 내 개별 명사와 경제/금융 용어 간 유사도가 높은 상위 20개 키워드 추출 """
        text = self.load_data()
        nouns = self.extract_nouns_Kiwi(text)
        noun_embeddings = self.create_embeddings(nouns)

        # 중복 방지를 위한 집합 생성
        unique_terms = set()
        similar_terms = []

        # 각 명사와 경제/금융 용어 사전 임베딩 간 유사도 계산
        for i, noun_embedding in enumerate(noun_embeddings):
            cosine_scores = util.cos_sim(noun_embedding.unsqueeze(0), self.economic_embeddings)[0]
            max_similarity = max(cosine_scores).item()

            # 유사도가 임계값 이상인 경우 해당 명사를 추가
            ## 추가 태스크2 : 임계값 조절해보면서 가장 적절하게 경제, 금융 용어를 뽑아내는 구간 찾기 
            if max_similarity > 0.3 and nouns[i] not in unique_terms:
                similar_terms.append((nouns[i], max_similarity))
                unique_terms.add(nouns[i])  # 중복 방지 집합에 추가

        # 유사도에 따라 상위 20개 키워드만 선택
        similar_terms = sorted(similar_terms, key=lambda x: x[1], reverse=True)[:20]
        return [term[0] for term in similar_terms]

    
    def ext_finan_terms_with_custom_similarity(self, metric="cosine"):
        """ 지정한 유사도 측정 방법을 사용하여 경제/금융 관련 상위 10개 키워드 추출 """
        text = self.load_data()
        nouns_text = ' '.join(self.extract_nouns_Kiwi(text))

        # KeyBERT로 키워드 추출
        keywords = self.keyBERT_model.extract_keywords(
            nouns_text, 
            keyphrase_ngram_range=(1, 1),
            stop_words=None,
            top_n=30,  # 더 많은 키워드를 추출하여 필터링
            use_maxsum=True,
            use_mmr=True,
            diversity=0.7
        )

        # print("KeyBERT로 추출된 원본 키워드:", keywords)
        keywords = [kw[0] for kw in keywords if isinstance(kw, tuple) and len(kw) > 0]
        filtered_keywords = []

        for kw in keywords:
            kw_embedding = self.create_embeddings([kw])
            
            # 선택한 유사도 측정 방법에 따라 점수 계산
            scores = []
            for econ_embedding in self.economic_embeddings:
                if metric == "euclidean":
                    scores.append(-self.euclidean_distance(kw_embedding.squeeze(0), econ_embedding).item())  # 유클리디안 거리 (음수 처리로 가까운 순)
                elif metric == "manhattan":
                    scores.append(-self.manhattan_distance(kw_embedding.squeeze(0), econ_embedding).item())  # 맨해튼 거리 (음수 처리)
                elif metric == "dot":
                    scores.append(self.dot_product(kw_embedding.squeeze(0), econ_embedding).item())  # 점곱
                else:  # 기본 코사인 유사도
                    scores.append(util.cos_sim(kw_embedding, econ_embedding).item())
            
            # 유사도 또는 거리 기준 상위 키워드 필터링
            if metric in ["euclidean", "manhattan"]:
                min_score = min(scores)
                if min_score < -0.6:  # 거리 임계값
                    filtered_keywords.append((kw, -min_score))
            else:
                max_score = max(scores)
                if max_score > 0.6:  # 유사도 임계값
                    filtered_keywords.append((kw, max_score))

        # 점수 기준 상위 10개 키워드 선택
        filtered_keywords = sorted(filtered_keywords, key=lambda x: x[1], reverse=True)[:10]
        filtered_keywords = [kw[0] for kw in filtered_keywords]

        # print(f"{metric} 방식을 통해 필터링된 상위 10개 키워드:", filtered_keywords)
        return filtered_keywords

def main():
    filepath = "data/articles/sample2.html"  # 파일 경로
    extractor = Ext_Key(filepath)
    
    similar_terms = extractor.ext_finan_terms()
    print("단순 유사도를 통한 상위 20개 경제/금융 키워드:", similar_terms)
    print('\n')

    financial_terms_cosine = extractor.ext_finan_terms_with_custom_similarity(metric="cosine")
    print("코사인 유사도를 통한 상위 10개 경제/금융 키워드:", financial_terms_cosine)
    print('\n')

    financial_terms_euclidean = extractor.ext_finan_terms_with_custom_similarity(metric="euclidean")
    print("유클리디안 거리를 통한 상위 10개 경제/금융 키워드:", financial_terms_euclidean)
    print('\n')
    
    financial_terms_manhattan = extractor.ext_finan_terms_with_custom_similarity(metric="manhattan")
    print("맨해튼 거리를 통한 상위 10개 경제/금융 키워드:", financial_terms_manhattan)
    print('\n')
    
    financial_terms_dot = extractor.ext_finan_terms_with_custom_similarity(metric="dot")
    print("점곱을 통한 상위 10개 경제/금융 키워드:", financial_terms_dot)

if __name__ == '__main__':
    main()
