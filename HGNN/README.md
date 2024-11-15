# Financial Keyword Analysis with HGNN
경제/금융 도메인의 키워드 관계를 분석하기 위한 하이퍼그래프 신경망(HGNN) 구현

## 소개
이 프로젝트는 경제/금융 도메인의 키워드들 간의 복잡한 관계를 하이퍼그래프 신경망을 통해 분석합니다. 단순한 키워드 쌍이 아닌, 여러 키워드 간의 관계를 동시에 고려할 수 있습니다.

### 주요 기능
- 키워드 간 관계 분석
- 관계 타입별 중요도 학습
- 유사 키워드 검색
- 관계 예측
- 시각화 도구 제공

## 설치 방법
```bash
# 1. 저장소 복제
git clone {저장소 URL}
cd financial-hgnn

# 2. 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# 3. 필요한 패키지 설치
pip install -r requirements.txt
```

## 사용 방법

### 기본 실행
```bash
python main.py
```

### 사용자 정의 데이터로 실행
```bash
python main.py --data_path your_data.json
```

### 주요 매개변수
- `--hidden_dim`: 은닉층 차원 (기본값: 128)
- `--epochs`: 학습 에폭 수 (기본값: 150)
- `--lr`: 학습률 (기본값: 0.0002)
- `--device`: 실행 디바이스 (cuda/cpu)

## 데이터 형식
사용자 정의 데이터는 다음과 같은 JSON 형식을 따라야 합니다:

```json
{
    "keywords": ["키워드1", "키워드2", ...],
    "relations": [
        {
            "keywords": ["키워드1", "키워드2", "키워드3"],
            "relation_type": "INFLUENCE",
            "weight": 1.0
        },
        ...
    ]
}
```

## 결과 예시

### 유사 키워드 분석
```
'통화정책'와 유사한 키워드들:
  - 중앙은행     : 0.8715
  - 금리       : 0.8258
  - 시중은행     : 0.6789
```

### 시각화
- 학습 과정 그래프
- 관계 네트워크
- 키워드 클러스터링

## 요구사항
- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib
- NetworkX

## 주의사항
- GPU 사용 시 CUDA 지원 확인 필요
- 대규모 데이터셋의 경우 충분한 메모리 필요

## 라이선스
MIT License