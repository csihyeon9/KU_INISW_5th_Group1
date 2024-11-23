## **뉴스 추천을 위한 하이퍼그래프 신경망 (Hypergraph Neural Networks for News Recommendation)**

### **소개**
고려대학교 INISW 아카데미 5기 S. H. Kwan, K. H. Kim, S. H. Park, J. H. Lee, C. H. Cho, S. H. Cha가 개발한 프로젝트입니다.  
이 프로젝트는 하이퍼그래프 신경망(HGNN)을 활용하여 뉴스 문서 간의 관계를 모델링하고, 개인화된 뉴스 추천 시스템을 제공합니다.

---

### **사용 방법**

1. **Python 가상 환경 설정**
   - 가상 환경을 생성하고 활성화합니다.

2. **필요 패키지 설치**
   ```bash
   pip install -r requirements.txt
   ```

3. **모델 학습 및 평가**
   - 뉴스 추천을 위한 HGNN 모델을 학습하고 평가하려면 다음 명령어를 실행하세요:
   ```bash
   python train.py
   ```

4. **개인화 추천**
   - 새로 추가된 문서(`./personalization_data/New.json`)에 기반하여 개인화 추천을 수행하려면:
   ```bash
   python recommendation.py
   ```

---

### **코드 설명**

#### **train.py**
`train.py`는 HGNN 모델 학습을 담당합니다.

1. **데이터 로드 및 전처리**
   - JSON 형식의 뉴스 데이터셋(`EconFinNews_Raw.json`)을 로드합니다.
   - 문서의 키워드를 기반으로 특성(feature), 레이블(label), 학습-테스트 분할 데이터를 생성합니다.

2. **하이퍼그래프 생성**
   - 키워드 공유 정보를 기반으로 문서 간의 관계를 모델링한 하이퍼그래프를 생성합니다.
   - **하이퍼그래프 행렬 \( H \)**와 **정규화된 그래프 행렬 \( G \)**를 생성하여 HGNN 학습에 사용합니다.

3. **HGNN 학습**
   - **모델 구성**: `HGNNRecommender`를 사용하여 문서 간 관계를 학습.
   - **학습 과정**:
     - 손실 함수: 교차 엔트로피(Cross-Entropy).
     - 최적화 방법: AdamW.
     - 학습률 조정: CosineAnnealingLR.
   - **결과**:
     - 학습이 완료되면 모델이 `hgnn_recommender_model.pth`로 저장됩니다.

---

#### **recommendation.py**
`recommendation.py`는 학습된 HGNN 모델을 활용하여 개인화된 추천을 수행합니다.

1. **모델 초기화**
   - `train.py`에서 저장된 학습된 모델(`hgnn_recommender_model.pth`)을 로드합니다.
   - 기존 문서 임베딩을 사전에 계산합니다.

2. **새 문서 추가**
   - `./personalization_data/New.json`에서 새 문서를 로드합니다.
   - 새 문서를 기존 하이퍼그래프에 통합하고, 하이퍼그래프를 업데이트합니다.
   - HGNN을 사용하여 새 문서의 임베딩을 계산합니다.

3. **추천 수행**
   - **코사인 유사도**:
     - 새 문서와 기존 문서 임베딩 간의 코사인 유사도를 계산합니다.
     - 유사도가 높은 상위 `k`개의 문서를 추천합니다.
   - **관련 키워드 추출**:
     - 추천된 문서의 키워드를 집계하여 새 문서와 관련된 키워드 목록을 생성합니다.

4. **출력**
   - 추천된 문서의 URL, 유사도, 키워드를 출력합니다.
   - 새 문서와 관련된 키워드를 출력합니다.

---

### **입출력 예시**

#### **입력**
1. **기존 데이터셋**: `EconFinNews_Raw.json`
   - 뉴스 기사 제목, 키워드, URL 등을 포함.
2. **새로운 문서(New.json)**:
   ```json
   [
       {
           "url": "https://n.news.naver.com/mnews/article/003/0012920303",
           "relation_type": "생활경제",
           "keywords": [
               "비트코인",
               "가상자산",
               "알트코인"
           ]
       }
   ]
   ```

#### **출력**
**콘솔 출력 예시**:
```plaintext
Processing new article: https://n.news.naver.com/mnews/article/003/0012920303

Top recommendations:
1. URL: https://n.news.naver.com/mnews/article/015/0005048226, Similarity: 0.1994, Keywords: 스타벅스, 매출 감소, 주당 순이익
2. URL: https://n.news.naver.com/mnews/article/088/0000911235, Similarity: 0.1811, Keywords: 삼성전자, 이재용, 조직 관료화

Related Keywords:
스타벅스, 매출 감소, 삼성전자, 주당 순이익
```

---

### **추천 시스템 설명**

1. **키워드 기반 하이퍼그래프 생성**
   - 문서의 키워드를 기반으로 하이퍼엣지를 생성하여 문서 간의 관계를 모델링합니다.
   - 키워드를 공유하는 문서들은 하이퍼엣지로 연결됩니다.

2. **Hypergraph Neural Network (HGNN) 학습**
   - HGNN은 하이퍼그래프의 구조를 학습하여 문서의 고차원 임베딩을 생성합니다.
   - 학습된 임베딩은 문서 간의 유사도를 계산하는 데 사용됩니다.

3. **코사인 유사도를 활용한 추천**
   - 새 문서의 임베딩을 기존 문서 임베딩과 비교하여 가장 유사한 문서를 추천합니다.
   - 유사도 점수가 높은 순서대로 상위 `k`개의 문서를 출력합니다.

4. **관련 키워드 추천**
   - 추천된 문서들의 키워드를 집계하여 새 문서와 연관된 키워드 목록을 생성합니다.

---

### **인용**

```plaintext
@article{feng2018hypergraph,
  title={Hypergraph Neural Networks},
  author={Feng, Yifan and You, Haoxuan and Zhang, Zizhao and Ji, Rongrong and Gao, Yue},
  journal={AAAI 2019},
  year={2018}
}
```

---

### **라이선스**
본 코드는 MIT 라이선스에 따라 공개되며, 자세한 내용은 LICENSE 파일을 참조하세요.
