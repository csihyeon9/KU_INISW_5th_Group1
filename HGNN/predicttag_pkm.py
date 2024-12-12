# predicttag_pkm.py

import sys
import os
from typing import Dict, List, Tuple, Set, Any
import json
import re
import numpy as np
from collections import defaultdict
from predicttag_sol import SingleArticleProcessor

class PKMProcessor:
    """
    PKM 데이터와 새로운 기사 데이터를 통합하는 클래스
    
    1. 기사 데이터의 키워드들 간 관계 매핑된 정보
    2. 기존 PKM 속에 쌓여있던, 서로 다른 관계로 매핑된 키워드들
    3. 이 둘을 다시 새로운 관계로 묶어야 해!
    
    -> How? 해당 키워드들이 등장한 각 기사들 속 동시 출현 빈도를 고려한 pmi 계산으로..
    -> 서로 다른 기사들 속 가장 높은 pmi 값을 가진 놈(키워드1, 키워드2)을 추려내고, 
       이 두 키워드들에 대해 hgnn 모델로 관계를 매핑
    """
    
    def __init__(self, model_path: str, pmi_path: str):
        """PKM 프로세서 초기화"""
        self.article_processor = SingleArticleProcessor(model_path, pmi_path)

    def calculate_pmi(self, new_article_data: Dict, pkm_data: Dict, kw1: str, kw2: str) -> float:
        """
        두 기사 간 키워드 쌍의 PMI 계산
        
        PMI(kw1, kw2) = log(P(kw1, kw2) / (P(kw1) * P(kw2)))
        
        여기서:
        - P(kw1) = kw1의 총 출현 횟수 / 모든 키워드의 총 출현 횟수
        - P(kw2) = kw2의 총 출현 횟수 / 모든 키워드의 총 출현 횟수
        - P(kw1, kw2) = 두 키워드의 공동 출현 횟수 / 모든 키워드의 총 출현 횟수
        """
        def count_keyword_occurrences(text: str, keyword: str) -> int:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            return len(re.findall(pattern, text))
        
        # 새 기사 텍스트
        new_text = new_article_data[0].get('contents', '')
        
        # PKM 기사 텍스트들 결합
        pkm_texts = []
        for title, data in pkm_data.items():
            if isinstance(data, dict) and 'contents' in data:
                pkm_texts.append(data['contents'])
        
        # 키워드 출현 횟수 계산
        kw1_count = count_keyword_occurrences(new_text, kw1)
        kw2_count = count_keyword_occurrences(new_text, kw2)
        
        for text in pkm_texts:
            kw1_count += count_keyword_occurrences(text, kw1)
            kw2_count += count_keyword_occurrences(text, kw2)
        
        # 공동 출현 횟수는 각 문서에서의 최소 출현 횟수의 합
        joint_count = min(
            count_keyword_occurrences(new_text, kw1),
            count_keyword_occurrences(new_text, kw2)
        )
        for text in pkm_texts:
            joint_count += min(
                count_keyword_occurrences(text, kw1),
                count_keyword_occurrences(text, kw2)
            )
        
        # 전체 단어 수 계산
        total_words = len(new_text.split()) + sum(len(text.split()) for text in pkm_texts)
        
        # 확률 계산 (스무딩 적용)
        p_kw1 = (kw1_count + 1e-10) / total_words
        p_kw2 = (kw2_count + 1e-10) / total_words
        p_joint = (joint_count + 1e-10) / total_words
        
        return float(np.log(p_joint / (p_kw1 * p_kw2)))

    def find_related_keywords(self, new_article_data: Dict, pkm_data: Dict, 
                            top_k: int = 10, pmi_threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """
        서로 다른 문서의 키워드 간 관련성이 높은 상위 쌍 찾기
        """
        # PKM 문서의 키워드 집합
        pkm_keywords = set()
        for relations in pkm_data.values():
            for relation in relations:
                pkm_keywords.update(relation['keywords'])
        
        # 새 문서의 키워드 집합
        new_keywords = set()
        for relation in new_article_data[0]['relations']:
            new_keywords.update(relation['keywords'])
        
        # 서로 다른 문서의 키워드 쌍에 대해서만 PMI 계산
        cross_doc_pairs = []
        for pkm_kw in pkm_keywords:
            for new_kw in new_keywords:
                pmi = self.calculate_pmi(new_article_data, pkm_data, pkm_kw, new_kw)
                if pmi >= pmi_threshold:
                    cross_doc_pairs.append((pkm_kw, new_kw, pmi))
        
        # PMI 값으로 정렬하여 상위 K개 반환
        return sorted(cross_doc_pairs, key=lambda x: x[2], reverse=True)[:top_k]

    def integrate_with_pkm(self, new_article_data: Dict, pkm_data: Dict) -> Dict:
        """새로운 기사를 PKM과 통합"""
        # 1. 새 기사 관계 분석
        integrated_data = pkm_data.copy()
        article_title = new_article_data[0]['title']
        new_relations = self.article_processor.process_article_relations(new_article_data)
        integrated_data.update(new_relations)
        
        # 2. 서로 다른 문서의 키워드 간 관계 찾기
        cross_doc_pairs = self.find_related_keywords(new_article_data, pkm_data, 
                                                   top_k=10, pmi_threshold=0.5)
        
        # 3. 교차 문서 관계 매핑
        cross_document_relations = {}
        test_data = [{"title": "test", "relations": []}]
        
        for pkm_kw, new_kw, pmi in cross_doc_pairs:
            # 관계 예측
            test_data[0]["relations"] = [{
                "verb": "관계",
                "keywords": [pkm_kw, new_kw]
            }]
            prediction = self.article_processor.process_article_relations(test_data)["test"][0]
            
            # 관계 정보 저장
            relation_key = f"{pkm_kw}_{new_kw}"
            cross_document_relations[relation_key] = {
                "source": pkm_kw,
                "target": new_kw,
                "category": prediction["category"],
                "confidence": prediction["confidence"],
                "pmi_strength": float(pmi),
                "source_article": list(pkm_data.keys())[0],
                "target_article": article_title
            }
        
        if cross_document_relations:
            integrated_data["cross_document_relations"] = cross_document_relations
        
        return integrated_data

def main():
    """메인 실행 함수"""
    try:
        print("PKM 통합 처리 시작")
        print("-" * 50)
        
        # 1. 경로 설정
        paths = {
            'new_article': 'data/pkm/test_article.json',
            'pkm': 'data/pkm/pkm_example.json',
            'model': 'results/ver2/models/ver1/hgnn_model.pth',
            'pmi': 'data/pairwise_pmi_values3.json',
            'output': 'data/pkm/final_integrated_pkm.json'
        }
        
        # 2. 입력 파일 존재 확인
        for key, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required {key} file not found at: {path}")
        
        # 3. 데이터 로드
        with open(paths['new_article'], 'r', encoding='utf-8') as f:
            new_article_data = json.load(f)
        with open(paths['pkm'], 'r', encoding='utf-8') as f:
            pkm_data = json.load(f)
            
        print(f"새로운 기사: {new_article_data[0]['title']}")
        print(f"기존 PKM 문서 수: {len(pkm_data)}")
        
        # 4. PKM 프로세서 초기화 및 실행
        processor = PKMProcessor(paths['model'], paths['pmi'])
        integrated_data = processor.integrate_with_pkm(
            new_article_data,
            pkm_data
        )
        
        # 5. 결과 저장
        with open(paths['output'], 'w', encoding='utf-8') as f:
            json.dump(integrated_data, f, ensure_ascii=False, indent=4)
        
        # 6. 결과 요약 출력
        article_count = len(integrated_data) - (1 if 'cross_document_relations' in integrated_data else 0)
        
        print("\n처리 완료:")
        print(f"- 통합된 문서 수: {article_count}")
        print(f"- 문서 내 관계 수: {sum(len(relations) for relations in integrated_data.values() if isinstance(relations, list))}")
        
        # 7. 문서 간 관계 출력
        if 'cross_document_relations' in integrated_data:
            cross_relations = integrated_data['cross_document_relations']
            print(f"\n문서 간 관계 수: {len(cross_relations)}")
            print("\n상위 문서 간 키워드 관계:")
            print("-" * 50)
            
            for key, rel in cross_relations.items():
                print(f"[{rel['source']} → {rel['target']}]")
                print(f"카테고리: {rel['category']}")
                print(f"PMI 강도: {rel['pmi_strength']:.3f}")
                print(f"신뢰도: {rel['confidence']:.3f}")
                print(f"출처: {rel['source_article']}")
                print(f"대상: {rel['target_article']}")
                print("-" * 50)
        
        print(f"\n결과 저장 위치: {paths['output']}")
        print("-" * 50)
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()


