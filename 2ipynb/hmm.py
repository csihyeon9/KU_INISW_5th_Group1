from koalanlp.Util import initialize, finalize
from koalanlp.proc import Tagger
from koalanlp import API

# KoalaNLP 초기화
initialize(java_options="-Xmx4g", KKMA="LATEST")  # KKMA 형태소 분석기 사용

def extract_subject_object_relation_with_koalanlp(text):
    """KoalaNLP를 사용한 주어, 목적어, 관계 추출"""
    tagger = Tagger(API.KKMA)  # KKMA 분석기 생성
    results = []

    # 문장을 형태소 분석
    for sentence in tagger(text):
        subject = None
        object_ = None
        relation = None

        # 형태소 단위로 주어, 목적어, 관계 추출
        for word in sentence.words:
            if word.postag.startswith("N") and subject is None:  # 첫 번째 명사를 주어로 간주
                subject = word.surface
            elif word.postag.startswith("V") and relation is None:  # 첫 번째 동사를 관계로 간주
                relation = word.surface
            elif word.postag.startswith("N") and object_ is None and subject is not None:
                object_ = word.surface  # 두 번째 명사를 목적어로 간주

        # 결과 저장
        if subject or object_ or relation:
            results.append({
                "sentence": sentence.surfaceString,
                "subject": subject,
                "object": object_,
                "relation": relation
            })

    return results

# 입력 텍스트
text = """
이례적인 11월 폭설이 내리면서 손해보험사들의 자동차 사고 접수가 1년 만에 60% 넘게 급증했다.
28일 손해보험업계에 따르면 삼성화재와 현대해상, KB손해보험 등 3개사의 전날 차량 사고 접수 건수는
1만8556건으로 지난해 11월 일평균(1만1138건)보다 66.6% 늘었다.
긴급출동 건수는 4만8801건으로 지난해 11월 일평균(3만7409건)보다 30.5% 증가했다.
"""

# KoalaNLP를 사용한 결과 추출
extracted_info = extract_subject_object_relation_with_koalanlp(text)

# 결과 출력
for info in extracted_info:
    print(f"문장: {info['sentence']}")
    print(f"주어: {info['subject']}")
    print(f"목적어: {info['object']}")
    print(f"관계: {info['relation']}")
    print("-" * 30)

# KoalaNLP 자원 정리
finalize()
