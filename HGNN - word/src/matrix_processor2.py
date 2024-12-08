import os
import torch
import numpy as np

def save_data_for_hgnn(output_path, pmi_matrix, incidence_matrix, G, keyword_to_idx, labels):
    # 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 데이터 저장
    torch.save({
        'X': torch.tensor(pmi_matrix, dtype=torch.float32),
        'H': torch.tensor(incidence_matrix, dtype=torch.float32),
        'G': torch.tensor(G, dtype=torch.float32),
        'keyword_to_idx': keyword_to_idx,
        'labels': torch.tensor(labels, dtype=torch.long)
    }, output_path)
    print(f"HGNN 학습 데이터를 {output_path}에 저장했습니다.")

def main():
    pairwise_pmi_path = "data/pairwise_pmi_values3.json"
    unique_keywords_path = "data/updated_unique_keywords.json"
    output_path = "data/processed_data/hgnn_data2.pt"

    # 데이터 처리 과정 (생략)
    pmi_matrix = np.random.rand(100, 100)  # 예제 데이터
    incidence_matrix = np.random.randint(2, size=(100, 20))
    G = np.random.rand(100, 100)  # 예제 데이터
    keyword_to_idx = {f"keyword_{i}": i for i in range(100)}
    labels = np.random.randint(0, 10, size=100)

    save_data_for_hgnn(output_path, pmi_matrix, incidence_matrix, G, keyword_to_idx, labels)

if __name__ == "__main__":
    main()
