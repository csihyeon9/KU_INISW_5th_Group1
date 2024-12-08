아래는 **초기 코드 수정 과정**부터 **추천 시스템 구현**까지의 모든 과정을 포함한 `README.md` 파일입니다.

---

### **`README.md`**

```markdown
# HGNN 추천 시스템

이 프로젝트는 하이퍼그래프 신경망(HGNN)을 활용하여 데이터의 고차원 관계를 학습하고, 키워드 추천을 제공하는 시스템입니다. 프로젝트 진행 중에는 초기 코드 수정, 모델 학습, 추천 시스템 구현까지 다양한 작업이 포함되었습니다.

---

## **프로젝트 구조**

```
HGNN_Recommendation/
├── config/
│   ├── config.yaml            # 모델, 학습 설정 및 데이터 경로 구성 파일
├── data/
│   ├── dataset/               # 원시 데이터 디렉토리
│   ├── processed_data/        # 학습용 데이터 디렉토리
│   ├── pairwise_pmi_values.json  # 하이퍼엣지 가중치 계산을 위한 PMI 값
├── models/
│   ├── HGNN_model.py          # HGNN 모델 구현
│   ├── layers.py              # HGNN을 위한 커스텀 레이어
├── src/
│   ├── utils/                 # 유틸리티 스크립트
│   │   ├── data_loader.py     # 데이터 로드 및 전처리
│   ├── visualization.py       # 학습 시각화
├── train3.py                  # 모델 학습 스크립트
├── recommendation.py          # 추천 시스템 스크립트
```

---

## **설치 방법**

### 요구 사항
- Python 3.8 이상
- 필요한 Python 라이브러리 설치:
  ```bash
  pip install -r requirements.txt
  ```
  예시 라이브러리:
  - `torch`
  - `numpy`
  - `scikit-learn`
  - `tqdm`
  - `seaborn`

---

## **초기 코드 수정 및 문제 해결**

### **1. 문제: 초기 코드가 정상적으로 작동하지 않음**
- **문제점**: 모델 정의, 데이터 크기 불일치, 희소 행렬 연산 등 다양한 오류 발생.
- **해결 과정**:
  1. **희소 행렬 슬라이싱 문제**:
     - PyTorch의 희소 행렬은 슬라이싱을 지원하지 않음.
     - 해결: 희소 행렬을 밀집 행렬로 변환한 후 슬라이싱, 이후 다시 희소 행렬로 변환.
  2. **입력 및 출력 크기 불일치**:
     - 저장된 모델과 현재 설정 파일 간의 `in_features` 크기 불일치.
     - 해결: 설정 파일 및 모델 정의를 저장된 모델과 일치하도록 수정.

### **2. 수정된 코드 적용**
- 학습 스크립트(`train3.py`) 수정:
  - 배치별로 `G_sparse`를 슬라이싱하고 희소 행렬로 복원.
  - 학습 완료 후 모델 저장 추가.
- 추천 스크립트(`recommendation.py`) 수정:
  - 저장된 모델 로드 및 입력 키워드 유효성 검증 추가.
  - 키워드 추천 알고리즘 구현.

---

## **실행 방법**

### 1. **데이터 준비**

#### **입력 데이터**
- 원시 데이터를 `data/dataset/` 디렉토리에 저장합니다.
- PMI 값은 `pairwise_pmi_values.json`에 저장되어야 합니다.

#### **데이터 전처리**
1. 데이터 전처리 스크립트를 실행하여 학습용 데이터를 생성합니다:
   ```bash
   python src/utils/data_loader.py
   ```
2. 전처리된 데이터는 `data/processed_data/hgnn_data2.pt`에 저장됩니다.

---

### 2. **모델 학습**

1. `config/config.yaml` 파일 설정:
   ```yaml
   model:
     in_features: 100  # 입력 특징 수
     hidden_features: 64
     out_features: 13  # 클래스 개수
   training:
     lr: 0.0001
     weight_decay: 0.0001
     epochs: 10
     batch_size: 32
     save_dir: "results/models"
     checkpoint_interval: 10
   ```

2. 모델 학습 실행:
   ```bash
   python train3.py
   ```

3. 출력 결과:
   - **체크포인트**: `results/models/hgnn_checkpoint_epoch_X.pth`
   - **최종 모델**: `results/models/hgnn_model.pth`
   - **학습 시각화**: `results/plots/training_history.png`

---

### 3. **추천 시스템**

#### 추천 시스템 실행
1. 추천 시스템 실행:
   ```bash
   python recommendation.py
   ```

2. 키워드를 입력하여 추천 결과 확인:
   ```plaintext
   추천을 원하시는 키워드를 입력하세요: 금융
   ```

3. 출력 예시:
   ```plaintext
   '금융'와 유사한 키워드 추천:
    - 경제: 유사도 0.82
    - 투자: 유사도 0.78
    - 시장: 유사도 0.75
   ```

---

## **주요 기능**

- **하이퍼그래프 신경망(HGNN):**
  - 데이터의 고차원 관계를 캡처하기 위한 그래프 기반 접근 방식.

- **키워드 추천:**
  - 학습된 임베딩과 코사인 유사도를 기반으로 상위 유사 키워드를 추천.

- **유연한 구성:**
  - `config.yaml`을 통해 모델 및 학습 과정을 쉽게 구성 가능.

---

## **문제 해결**

### **자주 발생하는 문제**
1. **Config KeyError**:
   - `in_features`, `hidden_features`, `out_features` 값이 데이터셋 및 모델과 일치하는지 확인하세요.

2. **데이터 경로 문제**:
   - `config.yaml` 파일의 경로 설정이 올바른지 확인하세요.

3. **특징 크기 불일치**:
   - `in_features` 값이 데이터셋의 입력 차원과 일치하는지 확인하세요.

### **FutureWarnings**:
PyTorch의 `torch.load` 경고를 제거하려면 다음 코드를 사용하세요:
```python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
```

---

## **추가 작업**

- 모델 성능 개선:
  - 추가 특징(feature)을 포함하거나 데이터 전처리 방법을 개선.
- 대규모 데이터셋 최적화:
  - 메모리 사용량을 줄이기 위해 희소 행렬 최적화.

---


### **코드 실행 순서**

#### **1. 데이터 준비**

1. **원시 데이터와 PMI 파일 준비**
   - 원시 데이터를 `data/dataset/` 디렉토리에 저장합니다.
   - 하이퍼엣지 가중치 계산을 위한 PMI 파일(`pairwise_pmi_values.json`)을 `data/` 디렉토리에 저장합니다.

2. **데이터 전처리 실행**
   - `data_loader.py`를 실행하여 원시 데이터를 처리하고 모델 학습에 필요한 데이터셋을 생성합니다:
     ```bash
     python src/utils/data_loader.py
     ```
   - 처리된 데이터는 `data/processed_data/hgnn_data2.pt`에 저장됩니다.

---

#### **2. 모델 학습**

1. **`config.yaml` 설정 파일 준비**
   - `config/config.yaml`에서 학습에 필요한 모델 구성, 데이터 경로, 학습 파라미터 등을 설정합니다.

   예시 설정:
   ```yaml
   model:
     in_features: 100
     hidden_features: 64
     out_features: 13

   training:
     lr: 0.0001
     weight_decay: 0.0001
     epochs: 10
     batch_size: 32
     save_dir: "results/models"
     checkpoint_interval: 5
   ```

2. **모델 학습 실행**
   - `train3.py`를 실행하여 모델을 학습합니다:
     ```bash
     python train3.py
     ```
   - 학습이 완료되면 다음 결과물이 생성됩니다:
     - 체크포인트: `results/models/hgnn_checkpoint_epoch_X.pth`
     - 최종 모델: `results/models/hgnn_model.pth`
     - 학습 시각화: `results/plots/training_history.png`

---

#### **3. 추천 시스템 실행**

1. **`recommendation.py` 실행**
   - 학습된 모델(`hgnn_model.pth`)과 전처리된 데이터(`hgnn_data2.pt`)를 사용하여 추천 시스템을 실행합니다:
     ```bash
     python recommendation.py
     ```

2. **키워드 입력**
   - 실행 중에 추천을 요청할 키워드를 입력합니다:
     ```plaintext
     추천을 원하시는 키워드를 입력하세요: 금융
     ```

3. **추천 결과 확인**
   - 입력된 키워드와 유사한 키워드를 출력합니다:
     ```plaintext
     '금융'와 유사한 키워드 추천:
      - 경제: 유사도 0.82
      - 투자: 유사도 0.78
      - 시장: 유사도 0.75
     ```

---

### **요약된 실행 순서**
1. **데이터 전처리**
   ```bash
   python src/utils/data_loader.py
   ```

2. **모델 학습**
   ```bash
   python train3.py
   ```

3. **추천 시스템 실행**
   ```bash
   python recommendation.py
   ```

---

### **추가 참고**
- 코드 실행 전 반드시 `config.yaml` 파일이 올바르게 설정되었는지 확인하세요.
- 데이터 전처리 → 모델 학습 → 추천 시스템의 순서를 반드시 지켜주세요.



HGNN/
|_config/
|         |_ __init__.py
|         |_ config.py
|         |_ config.yaml
|
|_data/
|     |_ processed_data/
|_ final_dataset3.json
|_ pairwise_pmi_values3.json
|_ updated_unique_keywords.json
|
|
|_models/
|          |_ __init__.py
|          |_ HGNN_model.py
|          |_ layers.py
|
|_results/
|         |_models/
|         |_plots/
|
|_src/
|     |_utils/
|     |          |_ data_loader.py
|     |          |_ graph_utils.py
|     |
|     |_ keyword_processor2.py
|     |_ matrix_processor2.py
|     |_ recommend.py
|     |_ train.py
|     |_ visualization.py
|
|_ train3.py