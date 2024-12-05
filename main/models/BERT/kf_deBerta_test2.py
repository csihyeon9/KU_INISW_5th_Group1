from transformers import AutoTokenizer, AutoModel
from keybert import KeyBERT
from kiwipiepy import Kiwi
from bs4 import BeautifulSoup
from sentence_transformers import util
import torch
import numpy as np
import json
from typing import List, Tuple, Set, Dict

import matplotlib.pyplot as plt

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

class Ext_Key:
    def __init__(self, filepath, dict_filepath):
        self.filepath = filepath

        self.tokenizer = AutoTokenizer.from_pretrained("upskyy/kf-deberta-multitask") # tokenizer 모델 
        self.model = AutoModel.from_pretrained("upskyy/kf-deberta-multitask")  # kf-deberta 기반 모델
        
        self.kiwi = Kiwi() # 형태소 분석기 

        # self.keyBERT_model = KeyBERT(model='multi-qa-mpnet-base-cos-v1')  # KeyBert 모델 => model = 'skt/kobert-base-v1', 'all-MIniLM-L6-v2'
        self.keyBERT_model = KeyBERT(model='distiluse-base-multilingual-cased-v1')

        # 기본 경제, 금융 용어 사전 + json 파일에서 용어 추가 
        self.base_economic_terms = list(ECONOMIC_TERMS)  # 경제, 금융 용어 사전
        self.additional_economic_terms = self.load_additional_terms(dict_filepath)
        
        # 기존, 경제, 금융 용어 사전 + 새로운 단어들에 대해 각자 벡터 임베딩 값 생성
        self.economic_embeddings = self.create_embeddings(self.base_economic_terms)
        self.economic_embeddings_plus = self.create_embeddings(self.additional_economic_terms)

        # 하나의 벡터 공간 내에 기존 + 새로운 단어들 벡터 임베딩 값 생성
        self.unique_economic_terms = list(set(self.base_economic_terms).union(set(self.additional_economic_terms)))
        self.unique_economic_embeddings = self.create_embeddings(self.unique_economic_terms)


    def load_additional_terms(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # JSON 구조에 따라 용어 추출
            if isinstance(data, dict) and "nodes" in data:
                terms = [node["word"] for node in data["nodes"] if "word" in node]
            elif isinstance(data, list):
                terms = [term["word"] if isinstance(term, dict) and "word" in term else term for term in data]
            else:
                print(f"Warning: Unexpected JSON structure in {filepath}")
                return []   
            # 용어 정제
            cleaned_terms = [term.strip() for term in terms if term and isinstance(term, str)]
            print(f"{filepath}에서 경제, 금융 용어 {len(cleaned_terms)}개 추가 완료")
            return cleaned_terms

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
        """ 기사 내 명사와 경제/금융 용어 사전 내 유사도가 높은 상위 20개 키워드 추출 """
        text = self.load_data()
        nouns = self.extract_nouns_Kiwi(text)
        noun_embeddings = self.create_embeddings(nouns)
        
        # 중복 방지를 위한 집합 생성
        unique_terms = set()
        similar_terms = []

        # 각 명사와 경제/금융 용어 사전 임베딩 간 유사도 계산
        for i, noun_embedding in enumerate(noun_embeddings):
            # 두 사전(기본 및 추가된 경제/금융 용어 사전)에 대해 유사도 계산
            cosine_scores = util.pytorch_cos_sim(noun_embedding.unsqueeze(0), torch.cat([self.economic_embeddings, self.economic_embeddings_plus], dim=0))[0]
            
            # 최대 유사도 계산
            max_similarity = max(cosine_scores).item()

            # 중복되지 않고 유사도가 임계값을 넘는 경우에만 추가
            if max_similarity > 0.4 and nouns[i] not in unique_terms:
                similar_terms.append((nouns[i], max_similarity))
                unique_terms.add(nouns[i])  # 중복 방지
            
        # 유사도에 따라 상위 20개 키워드만 선택
        similar_terms = sorted(similar_terms, key=lambda x: x[1], reverse=True)[:20]
        return [term[0] for term in similar_terms]

    def ext_finan_terms_with_custom_similarity(self, metric="cosine"):
        """ KeyBert로 필터링 이후 + 유사도 측정 방법을 사용해 경제/금융 관련 상위 20개 키워드 추출 """
        text = self.load_data()
        nouns_text = ' '.join(self.extract_nouns_Kiwi(text))

        # KeyBERT로 키워드 추출
        keywords = self.keyBERT_model.extract_keywords(
            nouns_text, 
            keyphrase_ngram_range=(1, 1),
            stop_words=None,
            top_n=50,  # 더 많은 키워드를 추출하여 필터링
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
            combined_embeddings = torch.cat([self.economic_embeddings, self.economic_embeddings_plus], dim=0)

            for econ_embedding in combined_embeddings:  # torch.cat([self.economic_embeddings, self.economic_embeddings_plus], dim=0))[0]
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
                if min_score < -0.4:  # 거리 임계값
                    filtered_keywords.append((kw, -min_score))
            else:
                max_score = max(scores)
                if max_score > 0.4:  # 유사도 임계값
                    filtered_keywords.append((kw, max_score))

        # 점수 기준 상위 10개 키워드 선택
        filtered_keywords = sorted(filtered_keywords, key=lambda x: x[1], reverse=True)[:20]
        filtered_keywords = [kw[0] for kw in filtered_keywords]

        # print(f"{metric} 방식을 통해 필터링된 상위 10개 키워드:", filtered_keywords)
        return filtered_keywords
    
    def evaluate_terms(self, predicted_terms: List[str], true_terms: List[str]) -> Dict[str, float]:
            """
            예측된 용어와 실제 정답 용어를 비교하여 다양한 성능 지표를 평가하는 함수
            
            Args:
                predicted_terms: 모델이 추출한 경제/금융 용어 리스트
                true_terms: 실제 정답으로 제공된 경제/금융 용어 리스트
                
            Returns:
                다양한 성능 평가 지표를 포함한 딕셔너리
            """
            predicted_set = set(predicted_terms)
            true_set = set(true_terms)
            
            # 기본 지표 계산을 위한 값들
            tp = len(predicted_set.intersection(true_set))
            fp = len(predicted_set - true_set)
            fn = len(true_set - predicted_set)
            tn = len(ECONOMIC_TERMS - (predicted_set.union(true_set)))  # 전체 용어 사전에서 미포함된 단어 수
            
            # 1. 기본 지표 계산
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # 2. 추가 지표 계산
            # Specificity (TNR: True Negative Rate)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Accuracy
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            # Balanced Accuracy
            balanced_accuracy = (recall + specificity) / 2
            
            # Jaccard Similarity
            jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            
            # 3. 고급 지표 계산
            # Positive Predictive Value (PPV) = Precision
            ppv = precision
            
            # Negative Predictive Value (NPV)
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # Matthews Correlation Coefficient (MCC)
            mcc_numerator = (tp * tn) - (fp * fn)
            mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 1
            mcc = mcc_numerator / mcc_denominator
            
            return {
                # 기본 지표
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1_score": round(f1, 3),
                
                # 추가 지표
                "specificity": round(specificity, 3),
                "accuracy": round(accuracy, 3),
                "balanced_accuracy": round(balanced_accuracy, 3),
                "jaccard": round(jaccard, 3),
                
                # 고급 지표
                "ppv": round(ppv, 3),
                "npv": round(npv, 3),
                "mcc": round(mcc, 3),
                
                # 원본 값들
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn
            }

def plot_evaluation_metrics(metrics_dict: Dict[str, Dict[str, float]], plot_type: str = "basic") -> None:
    """
    성능 평가 지표를 시각화하는 함수
    
    Args:
        metrics_dict: 모델별 성능 평가 지표를 담은 딕셔너리
        plot_type: 시각화할 지표 유형 ("basic", "advanced", "all")
    """
    if plot_type == "basic":
        metrics_to_plot = ['precision', 'recall', 'f1_score']
    elif plot_type == "advanced":
        metrics_to_plot = ['specificity', 'accuracy', 'balanced_accuracy', 'jaccard']
    else:  # "all"
        metrics_to_plot = ['precision', 'recall', 'f1_score', 'specificity', 
                          'accuracy', 'balanced_accuracy', 'jaccard', 'mcc']
    
    models = list(metrics_dict.keys())
    x = np.arange(len(models))
    width = 0.8 / len(metrics_to_plot)
    
    fig, ax = plt.subplots(figsize=(15, 7))
    
    for i, metric in enumerate(metrics_to_plot):
        values = [metrics_dict[model][metric] for model in models]
        ax.bar(x + i*width, values, width, label=metric)
    
    ax.set_ylabel('Scores')
    ax.set_title(f'Model Performance Comparison - {plot_type.capitalize()} Metrics')
    ax.set_xticks(x + width * len(metrics_to_plot) / 2)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def main():
    article_filepath = "data/articles/sample2.html"
    dict_filepath = "data/gnn_15000.json"
    
    extractor = Ext_Key(article_filepath, dict_filepath)
    
    ground_truth_terms = [
        '비트코인', '가상자산', '미연방준비제도', '기준금리', '장세', '모멘텀', '순환매', '도미넌스'
    ]
    
    evaluation_results = {}
    
    # 기본 DB + 코사인 유사도
    predicted_terms_db = extractor.ext_finan_terms()
    evaluation_results['DB + Cosine'] = extractor.evaluate_terms(predicted_terms_db, ground_truth_terms)
    
    # 다양한 유사도 측정 방식
    similarity_metrics = ['cosine', 'euclidean', 'manhattan', 'dot']
    for metric in similarity_metrics:
        predicted_terms = extractor.ext_finan_terms_with_custom_similarity(metric=metric)
        evaluation_results[f'KeyBERT + {metric.capitalize()}'] = extractor.evaluate_terms(
            predicted_terms, ground_truth_terms
        )
        print(f"\n{metric.capitalize()} Similarity - 예측된 키워드:", predicted_terms)
        
        # 상세 평가 결과 출력
        print(f"\n{metric.capitalize()} Similarity 평가 결과:")
        for metric_name, score in evaluation_results[f'KeyBERT + {metric.capitalize()}'].items():
            if metric_name not in ['true_positives', 'false_positives', 'false_negatives', 'true_negatives']:
                print(f"{metric_name}: {score}")
    
    # 다양한 시각화 제공    
    print("\n전체 지표 시각화")
    plot_evaluation_metrics(evaluation_results, "all")

if __name__ == '__main__':
    main()