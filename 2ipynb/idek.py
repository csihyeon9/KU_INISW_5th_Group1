from konlpy.tag import Okt
import re

class KoreanRelationExtractor:
    def __init__(self):
        self.tagger = Okt()
        
    def split_sentences(self, text):
        sentences = re.split('[.!?]\s+', text)
        return [s.strip() + '.' for s in sentences if s.strip()]

    def find_main_verb(self, pos_tagged):
        # 주요 서술어 찾기 (보조 동사 제외)
        exclude_verbs = ['있다', '없다', '하다', '되다', '내리다']
        for i in range(len(pos_tagged)-1, -1, -1):
            word, tag = pos_tagged[i]
            
            # 조사나 보조 동사 건너뛰기
            if tag in ['Josa', 'Eomi']:
                continue
                
            if tag in ['Verb', 'Adjective']:
                if word not in exclude_verbs:
                    # 앞의 명사와 결합된 동사 체크
                    if i > 0 and pos_tagged[i-1][1].startswith('Noun'):
                        return pos_tagged[i-1][0] + word
                    return word
                
        return None

    def extract_relations(self, sentence):
        pos_tagged = self.tagger.pos(sentence, norm=False, stem=False)
        
        # 주어 추출
        subject = None
        subject_markers = ['이', '가', '은', '는']
        for i, (word, tag) in enumerate(pos_tagged):
            if tag == 'Josa' and word in subject_markers:
                # 주어구 추출
                subj_tokens = []
                j = i - 1
                while j >= 0 and (pos_tagged[j][1].startswith('Noun') or 
                                pos_tagged[j][1] in ['Adjective', 'Determiner']):
                    subj_tokens.insert(0, pos_tagged[j][0])
                    j -= 1
                if subj_tokens:
                    subject = ' '.join(subj_tokens)
                break
        
        # 목적어 추출
        object_ = None
        object_markers = ['을', '를']
        for i, (word, tag) in enumerate(pos_tagged):
            if tag == 'Josa' and word in object_markers:
                # 목적어구 추출
                obj_tokens = []
                j = i - 1
                while j >= 0 and (pos_tagged[j][1].startswith('Noun') or 
                                pos_tagged[j][1] in ['Adjective', 'Determiner']):
                    obj_tokens.insert(0, pos_tagged[j][0])
                    j -= 1
                if obj_tokens:
                    object_ = ' '.join(obj_tokens)
                break
        
        # 주요 서술어 추출
        relation = self.find_main_verb(pos_tagged)
        if relation:
            # 부정문 처리
            for i, (word, tag) in enumerate(pos_tagged):
                if word in ['않다', '못하다', '말다'] and i > 0:
                    relation = pos_tagged[i-1][0] + ' ' + word
        
        return {
            'sentence': sentence,
            'subject': subject,
            'object': object_,
            'relation': relation
        }

    def process_text(self, text):
        sentences = self.split_sentences(text)
        return [self.extract_relations(sentence) for sentence in sentences]

def main(text):
    extractor = KoreanRelationExtractor()
    results = extractor.process_text(text)
    
    for result in results:
        print(f"\n문장: {result['sentence']}")
        print(f"주어: {result['subject']}")
        print(f"목적어: {result['object']}")
        print(f"관계: {result['relation']}")

# 테스트
text = """
이례적인 11월 폭설이 내리면서 손해보험사들의 자동차 사고 접수가 1년 만에 60% 넘게 급증했다. 28일 손해보험업계에 따르면 삼성화재와 현대해상, KB손해보험 등 3개사의 전날 차량 사고 접수 건수는 1만8556건으로 지난해 11월 일평균(1만1138건)보다 66.6% 늘었다. 긴급출동 건수는 4만8801건으로 지난해 11월 일평균(3만7409건)보다 30.5% 증가했다.  전날 갑작스러운 폭설로 차가 고장난 경우가 늘고 결빙으로 인한 교통사고가 급격히 증가한 데 따른 결과로 분석된다. 기상청에 따르면 이날 오전 8시 기준 적설량은 용인 백암 47.5㎝, 수원 43.0㎝, 군포 금정 42.4㎝, 안양 만안 40.7㎝ 등이다.  삼성화재가 지난 2019년부터 2022년까지 기상관측 자료와 보험사에 접수된 교통사고를 분석한 결과 겨울철 눈이 온 날의 교통사고 발생 건수는 눈이 오지 않은 날보다 17.6% 많았다. 이에 따른 교통사고 처리 피해액도 하루 평균 69억2000만원이 더 컸다.
"""

main(text)