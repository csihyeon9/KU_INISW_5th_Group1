# GNN Problem Recommender

**GNN Problem Recommender는 Graph Neural Network (GNN)을 이용하여 사용자가 입력한 키워드와 관련된 문제를 추천하는 시스템입니다.
이 프로젝트는 PyTorch와 PyTorch Geometric을 기반으로 하고, 문제 데이터는 SQLite 데이터베이스로 관리됩니다.**

## 프로젝트 구조

```
gnn-problem-recommender/
│
├── data/                                  # 데이터 관련 폴더
│   ├── raw/                               # 원본 크롤링 데이터
│   ├── processed/                         # 전처리된 그래프 데이터와 임베딩 파일
│   │   ├── nodes_korean_final.json        # 노드 데이터
│   │   ├── edges_korean_final.json        # 엣지 데이터
│   │   ├── topic_mapping.pkl              # 라벨-주제 매핑 파일
│   │   ├── gnn_data.pkl                   # GNN 데이터
│   │   └── gnn_score/                     # 학습 과정 시각화 그래프 저장 폴더
│   │       ├── training_loss.png          # 학습 손실 시각화 그래프
│   │       ├── accuracy_over_epochs.png   # 정확도 시각화 그래프
│   │       └── f1_score_over_epochs.png   # F1 점수 시각화 그래프
│   └── recommendation_problems.db         # 문제 추천 데이터베이스
│
├── models/                                # 학습된 GNN 모델 저장 폴더
│   └── gnn_model.pth                      # GNN 모델 가중치 파일
│
├── notebooks/                             # 실험 및 분석 노트북
│   └── data_processing.ipynb              # 데이터 전처리 및 모델 실험 노트북
│
├── src/                                   # 소스 코드 폴더
│   ├── data_processing.py                 # 데이터 수집, 키워드 추출 및 그래프 변환 코드
│   ├── gnn_model.py                       # GraphSAGE 및 GCN 모델 학습 코드
│   ├── recommender.py                     # 문제 추천 및 협업 필터링 로직
│   └── visualization.py                   # 그래프 시각화 코드
│
├── initialize_database.py                 # 데이터베이스 초기화 스크립트
├── config.yaml                            # 설정 파일
├── requirements.txt                       # 필요한 라이브러리 목록
└── README.md                              # 프로젝트 설명 파일
```
## 진행 환경
>> Window 10 / Vscode


## 주요 파일 설명

- **data_processing.py**: `nodes_korean_final.json` 및 `edges_korean_final.json` 데이터를 불러와 그래프로 변환하는 코드입니다.
- **gnn_model.py**: GNN 모델(예: GCN)을 정의하고 학습하며, 학습된 모델을 저장하는 코드입니다.
- **recommender.py**: 학습된 모델을 로드하여 사용자의 키워드와 유사한 문제를 추천하는 시스템입니다.
- **visualization.py**: 학습된 GNN 그래프를 시각화하는 코드입니다.
- **recommendation_problems.db**: 문제 데이터베이스로, 문제 ID, 제목, 설명 및 각 문제의 임베딩을 포함합니다.

## 실행 방법

### 1. 환경 설정

필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

### 2. 데이터 전처리 및 그래프 생성

`data_processing.py`를 실행하여 데이터를 그래프 형식으로 변환하고, 노드와 엣지를 `processed/` 폴더에 저장합니다.

```bash
python src/data_processing.py
```

### 3. GNN 모델 학습

`gnn_model.py`를 실행하여 GNN 모델을 학습하고, 학습된 모델 가중치를 `models/gnn_model.pth` 파일로 저장합니다.

```bash
python src/gnn_model.py
```

### 4. 문제 추천 시스템 초기화

데이터베이스 초기화 파일 `initialize_database.py`를 작성하여 실행합니다. 이 파일은 `recommendation_problems.db`에 문제 및 임베딩 데이터를 삽입합니다.

```bash
python initialize_database.py
```

### 5. 문제 추천 시스템 실행

`recommender.py`를 실행하고, 추천받고자 하는 키워드를 명령줄 인자로 전달하여 문제를 추천받을 수 있습니다.

```bash
python src/recommender.py "Finance"
```

## 코드 설명

### gnn_model.py
- **GCN**: 2개의 GCNConv 레이어를 사용하여 그래프 데이터를 학습하는 그래프 신경망 모델입니다.
- **train()**, **evaluate()**: 모델 학습과 평가를 위한 함수입니다.
- **main()**: 모델 초기화 및 학습, 평가 결과 시각화 후 저장을 담당합니다.

### recommender.py
- **RecommenderSystem**: 키워드 임베딩을 사용하여 데이터베이스 내의 문제와 유사도를 계산해 관련성이 높은 문제를 추천하는 클래스입니다.
- **recommend_problems_by_keyword()**: 입력된 키워드와 유사한 문제를 추천합니다.

## 기여 방법

이 프로젝트에 기여하고 싶으신 분은 이 리포지토리를 포크하고, 풀 리퀘스트를 제출해 주세요. 문의 사항이 있으시면 언제든지 이슈를 남겨 주세요.

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
