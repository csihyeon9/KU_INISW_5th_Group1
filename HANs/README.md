# 실행 순서:
### 1. 가상환경 설정 및 의존성 설치
```bash
# 가상환경 생성
python -m venv hgan-env

# 가상환경 활성화
source hgan-env/bin/activate  # Linux/Mac
.\hgan-env\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 샘플 데이터셋 생성
```bash
python generate_sample_data.py
```

### 3. 모델 학습 실행
```bash
python main.py --data_path data/raw/financial_data.json
```

### 4. 선택적 기능:
- wandb 로깅 활성화: --wandb_project your_project_name
- tensorboard 사용: --use_tensorboard true
- GPU 지정: --device cuda:0