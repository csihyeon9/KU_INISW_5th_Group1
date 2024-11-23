# --------------------------------------------------------
# Utility functions for Hypergraph
#
# Author: Yifan Feng
# Date: November 2018
# --------------------------------------------------------
import numpy as np


def Eu_dis(x):
    """
    Calculate the distance among each row of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature.
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    """
    H = None
    for h in H_list:
        if h is not None and (isinstance(h, np.ndarray) and h.size > 0):  # 유효성 검사 추가
            if H is None:
                H = h
            else:
                if isinstance(h, list):  # 리스트 형식 처리
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
                else:
                    H = np.hstack((H, h))  # ndarray 처리
    if H is None:  # 결과가 비어 있으면 빈 배열 반환
        H = np.array([])
    return H


def generate_G_from_H(H):
    """Generate G matrix from hypergraph incidence matrix H"""
    D_v = np.sum(H, axis=1)
    D_e = np.sum(H, axis=0)

    D_v_inv = np.divide(1., np.sqrt(D_v), where=D_v != 0)
    D_e_inv = np.divide(1., np.sqrt(D_e), where=D_e != 0)

    G = np.diag(D_v_inv) @ H @ np.diag(D_e_inv) @ H.T @ np.diag(D_v_inv)

    return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    Construct hypergraph incidence matrix from distance matrix
    """
    n_obj = dis_mat.shape[0]
    H = np.zeros((n_obj, n_obj))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0  # 자신의 거리 제거
        dis_vec = np.array(dis_mat[center_idx]).squeeze()  # dis_vec이 1차원 배열인지 확인
        
        nearest_idx = np.argsort(dis_vec)[:k_neig]  # KNN 인덱스 선택
        avg_dis = np.average(dis_vec[nearest_idx])  # 평균 거리 계산
        
        if avg_dis == 0:
            print(f"Warning: Average distance is 0 for center_idx={center_idx}")
            continue
        
        for node_idx in nearest_idx:
            if node_idx >= n_obj or node_idx < 0:  # 유효하지 않은 인덱스 방지
                print(f"Warning: Invalid index {node_idx} for center_idx={center_idx}")
                continue
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H



def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from feature matrix
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if isinstance(K_neigs, int):
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if H_tmp is None or H_tmp.size == 0:
            print(f"Warning: H_tmp is empty for K_neig={k_neig}")
            continue
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H


def construct_H_with_keywords(documents, threshold=0.2):
    """Construct hypergraph incidence matrix from document keywords"""
    n_docs = len(documents)
    H = np.zeros((n_docs, n_docs))

    for i in range(n_docs):
        for j in range(n_docs):
            keywords_i = set(documents[i]['keywords'])
            keywords_j = set(documents[j]['keywords'])
            similarity = len(keywords_i.intersection(keywords_j)) / len(keywords_i.union(keywords_j))
            if similarity > threshold:
                H[i][j] = similarity

    return H
