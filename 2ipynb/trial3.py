import re
from konlpy.tag import Kkma

def analyze_sentence(text):
    kkma = Kkma()
    
    # 문장 분리
    sentences = kkma.sentences(text)
    
    analysis_results = []
    
    for sentence in sentences:
        # 형태소 분석 및 품사 태깅
        pos_tags = kkma.pos(sentence)
        
        # 주어 찾기 (더 포괄적인 방식)
        subjects = []
        for i, (word, tag) in enumerate(pos_tags):
            # 명사성 단어들 포함 (고유명사, 일반명사, 의존명사 등)
            if tag in ['NNG', 'NNP', 'NNB', 'NR']:
                # 길이가 1글자 이상이면 주어 후보
                if len(word) > 1:
                    subjects.append(word)
                # 2글자 이상 단어 중 조사와 연결된 경우도 고려
                if (i+1 < len(pos_tags)) and (pos_tags[i+1][0] in ['이', '가', '은', '는', '도']):
                    subjects.append(word)
        
        # 목적어 찾기 (목적격 조사가 있는 명사)
        objects = []
        for i, (word, tag) in enumerate(pos_tags):
            if tag in ['NNG', 'NNP', 'NNB', 'NR']:
                if (i+1 < len(pos_tags)) and (pos_tags[i+1][0] in ['을', '를']):
                    objects.append(word)
        
        # 관계 찾기 (동사, 형용사, 보조용언 포함)
        relations = [
            word for word, tag in pos_tags 
            if (tag.startswith('V') or tag.startswith('VA') or tag == 'VX')
            and len(word) > 1  # 1글자 단어 제외
        ]
        
        # 중복 제거 및 정리
        subjects = list(dict.fromkeys(subjects))
        objects = list(dict.fromkeys(objects))
        relations = list(dict.fromkeys(relations))
        
        analysis_results.append({
            'sentence': sentence,
            'subjects': subjects,
            'objects': objects,
            'relations': relations
        })
    
    return analysis_results

# 텍스트 입력
text = """이례적인 11월 폭설이 내리면서 손해보험사들의 자동차 사고 접수가 1년 만에 60% 넘게 급증했다. 28일 손해보험업계에 따르면 삼성화재와 현대해상, KB손해보험 등 3개사의 전날 차량 사고 접수 건수는 1만8556건으로 지난해 11월 일평균(1만1138건)보다 66.6% 늘었다. 긴급출동 건수는 4만8801건으로 지난해 11월 일평균(3만7409건)보다 30.5% 증가했다. 전날 갑작스러운 폭설로 차가 고장난 경우가 늘고 결빙으로 인한 교통사고가 급격히 증가한 데 따른 결과로 분석된다. 기상청에 따르면 이날 오전 8시 기준 적설량은 용인 백암 47.5㎝, 수원 43.0㎝, 군포 금정 42.4㎝, 안양 만안 40.7㎝ 등이다. 삼성화재가 지난 2019년부터 2022년까지 기상관측 자료와 보험사에 접수된 교통사고를 분석한 결과 겨울철 눈이 온 날의 교통사고 발생 건수는 눈이 오지 않은 날보다 17.6% 많았다. 이에 따른 교통사고 처리 피해액도 하루 평균 69억2000만원이 더 컸다."""

# 분석 실행
results = analyze_sentence(text)

# 결과 출력
for result in results:
    print("문장:", result['sentence'])
    print("주어:", result['subjects'])
    print("목적어:", result['objects'])
    print("관계:", result['relations'])
    print("---")