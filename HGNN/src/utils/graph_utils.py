from collections import defaultdict

def create_hyperedges(data, keyword_to_idx, category_to_idx):
    """
    하이퍼엣지를 생성하는 함수
    :param data: 원본 데이터
    :param keyword_to_idx: 키워드-인덱스 매핑 딕셔너리
    :param category_to_idx: 카테고리-인덱스 매핑 딕셔너리
    :return: 하이퍼엣지 딕셔너리
    """
    hyperedges = defaultdict(list)
    
    for article in data:
        relations = article.get('relations', {})
        for category, relation_list in relations.items():
            cat_idx = category_to_idx[category]
            for relation in relation_list:
                keywords_in_relation = []
                if len(relation) == 3:
                    keywords_in_relation = [word for word in relation[:2] if not word.endswith(("하다", "되다", "치다", "다"))]
                elif len(relation) == 2:
                    keywords_in_relation = [relation[1]]
                
                for keyword in keywords_in_relation:
                    if keyword in keyword_to_idx:
                        hyperedges[cat_idx].append(keyword_to_idx[keyword])
    
    return hyperedges