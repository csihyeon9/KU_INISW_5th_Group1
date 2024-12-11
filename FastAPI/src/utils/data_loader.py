import torch

def load_data(data_path):
    """
    전처리된 데이터를 로드하는 함수
    :param data_path: 데이터 파일 경로
    :return: 특성 행렬 X, 하이퍼그래프 인접 행렬 H, 레이블
    """
    data = torch.load(data_path)
    X = data['X']
    H = data['H']
    keyword_to_idx = data['keyword_to_idx']
    category_to_idx = data['category_to_idx']
    
    # 키워드별 카테고리 매핑 생성
    keyword_category = {}
    for article in data['raw_data']:
        for category, relations in article['relations'].items():
            for relation in relations:
                keywords = [word for word in relation if word in keyword_to_idx]
                for keyword in keywords:
                    keyword_category[keyword] = category_to_idx[category]
    
    # 레이블 생성
    labels = torch.zeros(len(keyword_to_idx), dtype=torch.long)
    for keyword, idx in keyword_to_idx.items():
        if keyword in keyword_category:
            labels[idx] = keyword_category[keyword]
    
    return X, H, labels