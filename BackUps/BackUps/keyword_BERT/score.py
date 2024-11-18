# score.py

import sys
import os
import matplotlib.pyplot as plt  # 그래프 시각화를 위한 라이브러리

# 현재 파일의 디렉토리를 모듈 경로에 추가하여 extractor.py 파일을 찾을 수 있도록 함
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extractor import Ext_Key

def evaluate_terms(predicted_terms, true_terms):
    """
    예측된 용어와 실제 정답 용어를 비교하여 성능을 평가하는 함수.
    정밀도(Precision), 재현율(Recall), F1-score를 계산함.
    
    :param predicted_terms: 모델이 추출한 경제/금융 용어 리스트
    :param true_terms: 실제 정답으로 제공된 경제/금융 용어 리스트
    :return: 성능 평가 지표 딕셔너리 (정밀도, 재현율, F1-score 포함)
    """
    # True Positive, False Positive, False Negative 계산
    tp = len(set(predicted_terms) & set(true_terms))  # 예측이 맞은 용어의 수
    fp = len(set(predicted_terms) - set(true_terms))  # 잘못 예측한 용어의 수
    fn = len(set(true_terms) - set(predicted_terms))  # 실제 정답에 있지만 예측하지 못한 용어의 수

    # Precision, Recall, F1 계산
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0

    return {"precision": precision, "recall": recall, "f1_score": f1}

def plot_metrics(metrics):
    """
    성능 평가 지표를 그래프로 시각화하는 함수.
    
    :param metrics: 정밀도, 재현율, F1-score가 포함된 성능 평가 지표 딕셔너리
    """
    # 지표의 이름과 값을 분리
    labels = list(metrics.keys())
    values = list(metrics.values())
    
    # 막대 그래프 생성
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['skyblue', 'salmon', 'lightgreen'])
    plt.ylim(0, 1)
    plt.title("Model Performance Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    
    # 각 막대에 값 표시
    for i, value in enumerate(values):
        plt.text(i, value + 0.02, f"{value:.2f}", ha='center', va='bottom')
    
    plt.show()

def main(filepath, true_terms):
    """
    HTML 파일에서 경제/금융 용어를 추출하고 평가하는 메인 함수.
    
    :param filepath: 분석할 HTML 파일의 경로
    :param true_terms: 실제 정답 용어 리스트
    """
    # Ext_Key 클래스를 사용하여 키워드 추출
    extractor = Ext_Key(filepath)
    predicted_terms = extractor.ext_finan_terms()
    
    # 성능 평가
    metrics = evaluate_terms(predicted_terms, true_terms)
    print("코사인 유사도를 통한 상위 20개 경제/금융 키워드: \n", predicted_terms)
    print("\n평가 지표:", metrics)
    
    # 성능 평가 지표 시각화
    plot_metrics(metrics)

if __name__ == '__main__':
    # 평가할 HTML 파일 경로와 정답 키워드 리스트 설정
    filepath = "./sample2.html"
    ground_truth_terms = [
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
    ]

    # main 함수 실행하여 평가 수행
    main(filepath, ground_truth_terms)
