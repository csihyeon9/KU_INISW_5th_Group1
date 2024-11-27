import os
import json
import numpy as np
import pickle
import torch
import argparse
import time
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model import *  # HGNN 모델을 정의한 파일로 가정
from util import Data, split_validation

# 데이터 준비 및 인접 행렬 생성
def build_adjacency_matrix(sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for session in sessions:
        unique_nodes = np.unique(session)
        length = len(unique_nodes)
        s = indptr[-1]
        indptr.append(s + length)
        for node in unique_nodes:
            indices.append(node - 1)
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(sessions), n_node))
    return matrix

# json_data 폴더에서 각 json 파일의 "content"를 불러옵니다.
def load_articles(json_folder_path):
    articles = []
    for filename in os.listdir(json_folder_path):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(json_folder_path, filename), 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    def extract_content(data):
                        if isinstance(data, dict):
                            return data.get("content", "")
                        elif isinstance(data, list):
                            contents = []
                            for item in data:
                                if isinstance(item, dict):
                                    contents.append(item.get("content", ""))
                            return " ".join(contents)
                        return ""
                    content = extract_content(json_data)
                    if content:
                        articles.append(content)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file {filename}")
            except Exception as e:
                print(f"Unexpected error processing file {filename}: {e}")
    return articles

# 데이터 전처리 함수
def preprocess_data(data):
    if not data:
        return []
    if not isinstance(data, list):
        data = [data]
    def flatten(x):
        if isinstance(x, (list, np.ndarray)):
            return [item for sublist in x for item in flatten(sublist)]
        return [x]
    data = [entry for entry in data if entry]
    return data

# 사전에 정의한 단어 리스트
keyword_list = [
    "블록체인", "비트코인", "암호화폐", "디지털 자산", "분산원장", "비트코인 채굴", "작업 증명", "지분 증명", "스마트 계약", "트랜잭션",
    "지갑", "토큰", "ICO", "거래소", "탈중앙화", "중앙화 거래소", "탈중앙화 거래소", "해시 함수", "가상화폐", "시총",
    "유동성", "공매도", "레버리지", "마진 거래", "선물 거래", "옵션 거래", "가스비", "스테이킹", "디파이", "이자 농사",
    "렌딩", "대출", "스왑", "라이트닝 네트워크", "노드", "메인넷", "테스트넷", "하드포크", "소프트포크", "비트코인 반감기",
    "체인 분석", "P2P 거래", "51% 공격", "네트워크 효과", "해시레이트", "멀티시그", "블록 보상", "세그윗", "블록 크기", "블록 높이",
    "메모풀", "검열 저항성", "컨센서스 알고리즘", "스캘러빌리티", "오라클", "블록 시간", "트랜잭션 속도", "유틸리티 토큰", "스테이블코인",
    "CBDC", "암호화폐 규제", "암호화폐 채택률", "플래시 론", "탈중앙화 자율 조직", "체인 분리", "체인 리조그", "데이터 무결성",
    "트러스트리스", "네트워크 해시레이트", "키 복구", "하드웨어 월렛", "핫 월렛", "콜드 월렛", "공공키 암호화", "프라이빗 키", "NFT",
    "알트코인", "비트코인 도미넌스", "암호화폐 거래량", "비트코인 가격 변동성", "암호화폐 시장", "OTC 거래", "거래 수수료", "암호화폐 투자",
    "비트코인 ETF", "금융 기술", "디지털 화폐", "블록체인 기술", "암호화폐 보안", "거래소 해킹", "암호화폐 사기", "암호화폐 채굴 장비", "비트코인 거래"
]

# 각 기사의 단어 출현 정보를 수집합니다.
def collect_word_indices(articles, keyword_list):
    sessions = []
    for article in articles:
        word_indices = []
        for idx, word in enumerate(keyword_list):
            if word in article:
                word_indices.append(idx + 1)  # 단어의 인덱스 기록 (1부터 시작)
        sessions.append(word_indices)
    return sessions

# 각 기사 내 단어 간 자카드 유사도 계산 및 인접 행렬 생성
def build_jaccard_similarity_matrix(sessions, n_node):
    matrix = np.zeros((n_node, n_node))
    for session in sessions:
        if len(session) == 0:
            continue  # 빈 세션은 건너뜁니다.
        for i in range(len(session)):
            for j in range(i + 1, len(session)):
                idx_i, idx_j = session[i] - 1, session[j] - 1
                set_i, set_j = set([idx_i]), set([idx_j])
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                if union == 0:
                    continue  # union이 0인 경우를 건너뜁니다.
                jaccard_similarity = intersection / union
                matrix[idx_i, idx_j] += jaccard_similarity
                matrix[idx_j, idx_i] += jaccard_similarity
    return csr_matrix(matrix)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--embSize', type=int, default=100, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--layer', type=float, default=3, help='the number of layer used')
    parser.add_argument('--beta', type=float, default=0.01, help='ssl task maginitude')
    parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')

    opt = parser.parse_args()
    print(opt)

    # JSON 데이터 로드
    json_folder_path = "json_data"
    articles = load_articles(json_folder_path)

    # 단어 인덱스 수집
    sessions = collect_word_indices(articles, keyword_list)
    (train_data, train_targets), (test_data, test_targets) = split_validation((sessions, sessions), valid_portion=0.2)

    # 노드 수 동적 결정
    n_node = len(keyword_list)
    print(f"Determined n_node: {n_node}")

    # 자카드 유사도 기반 인접 행렬 생성
    adjacency_matrix = build_jaccard_similarity_matrix(sessions, n_node)

    # 모델 학습 및 평가
    train_data = Data([train_data, train_targets], shuffle=True, n_node=n_node)
    test_data = Data([test_data, test_targets], shuffle=True, n_node=n_node)

    # 모델 초기화 및 학습
    model = trans_to_cuda(DHCN(adjacency=adjacency_matrix, n_node=n_node, lr=opt.lr, l2=opt.l2, beta=opt.beta,
                               layers=opt.layer, emb_size=opt.embSize, batch_size=opt.batchSize, dataset='custom'))

    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))

    # 최종 임베딩 출력 확인
    final_embeddings = model.get_embeddings()  # 모델에서 임베딩 추출
    print("Final embeddings shape:", final_embeddings.shape)

    # 4. 단어 간 유사도 계산
    similarity_matrix = cosine_similarity(final_embeddings)
    print("Similarity matrix between words:", similarity_matrix)

    # 5. t-SNE를 이용한 임베딩 시각화
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(final_embeddings)

    plt.figure(figsize=(15, 10))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    for i, word in enumerate(keyword_list):
        plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
    plt.title("Word Embeddings Visualization")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
