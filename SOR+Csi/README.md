# **금융 문장 분석 프로젝트**

주어진 텍스트 문서에서 금융 용어를 기반으로 문장을 분석하고, 문장 내 용어들 간의 관계(동사)를 추출하는 NLP. 
KoalaNLP와 JSON 기반 금융 용어 리스트를 사용하여 유연하고 확장 가능한 분석을 제공.

---

## **프로젝트 구조**

```
Realtion_extract_NLP/
│
├── data/
│   ├── article.txt           # 분석할 텍스트 파일
│   ├── financial_terms.json  # 금융 용어 JSON 파일
│   ├── not_verb.json  # 동사가 아닌 것 JSON 파일
│   ├── yes_verb.json  # 동사인 것 JSON 파일
│
├── src
│   ├── file_loader.py            # 텍스트 파일 로드
│   ├── sentence_splitter.py      # 문장 분리
│   ├── term_extractor.py         # 금융 용어 추출
│   ├── sentence_filter.py        # 문장 필터링
│   ├── relation_extractor.py     # 동사(관계) 추출
│
├── main.py                   # 메인 실행 파일
├── README.md                 # 프로젝트 설명 파일
├── requirements.txt
```

---

## **기능**

1. **file_loader.py**:
   - `article.txt` 파일에서 텍스트 불러오기

2. **sentence_splitter.py**:
   - 텍스트를 문장 단위로 분리

3. **term_extractor.py**:
   - JSON 파일(`financial_terms.json`)에서 금융 용어를 로드한 후, 각 문장에서 금융 용어를 추출

4. **sentence_filter.py**:
   - 금융 용어가 정확히 2개 포함된 문장만 선택

5. **relation_extractor.py**:
   - 문장에서 동사를 추출하고, 활용형으로 변환하여 더 이해하기 쉽게 출력

6. **main.py**:
   - 각 문장에 대해 추출된 금융 용어와 동사 관계를 출력

---

## **설치 및 실행**

### **1. 환경 설정**
Python 3.8 이상이 설치된 환경에서 실행을 권장합니다. (3.8 설치 후 가상환경에 py -3.8 -m venv ./이름)

### **2. 의존성 설치**
```bash
pip install -r requirements.txt
```

---

### **3. 데이터 준비**
- `data/article(1번부터 n번까지).txt`에 분석할 텍스트 파일을 추가
- `data/financial_terms.json`에 분석에 사용할 금융 용어 리스트를 작성
- `data/not_verb.json`에 분석에 사용할 동사가 아닌 것 리스트를 작성
- `data/yes_verb.json`에 분석에 사용할 동사인 것 리스트를 작성

---

### **4. 실행**
프로젝트 디렉토리에서 `main.py`를 실행합니다:
```bash
python main.py
```

---

## **출력 예시**
```plaintext
기사: article1.txt
문장: 범용 화폐인 달러를 최대한 수수료 없이 국내에서 바꿔 간 후 현지에서 환전하는 것이 수수료를 더 아낄 수 있는 경우가 꽤 된다.
금융 용어: ['달러', '수수료']
동사(관계): ['바꾸어', '아끼', '되ㄴ다', '대한']
```

made by. csihyeon9