import os
from src.file_loader import load_article
from src.sentence_splitter import split_sentences
from src.term_extractor import extract_terms
from src.sentence_filter import filter_sentences_by_term_count
from src.relation_extractor import extract_verbs
from koalanlp.Util import initialize, finalize
import json


def load_json_list(json_path, key):
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data.get(key, [])
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {json_path}")
        return []
    except json.JSONDecodeError:
        print(f"JSON 파일 형식이 올바르지 않습니다: {json_path}")
        return []


def process_file(file_path, financial_terms, not_verbs, yes_verbs):
    document = load_article(file_path)
    sentences = split_sentences(document)
    sentence_with_terms = [
        (sentence, extract_terms(sentence, financial_terms)) for sentence in sentences
    ]
    filtered_sentences = filter_sentences_by_term_count(sentence_with_terms, min_count=2, max_count=2)

    results = []

    for sentence, terms in filtered_sentences:
        verbs = extract_verbs(sentence, not_verbs, yes_verbs)  # yes_verbs 전달
        if verbs:  # 동사가 있는 경우만 처리
            results.append({
                "article": os.path.basename(file_path),
                "sentence": sentence,
                "terms": terms,
                "verbs": verbs
            })

    return results


def main():
    # KoalaNLP 초기화
    initialize(java_options="-Xmx4g --add-opens java.base/java.util=ALL-UNNAMED --add-opens java.base/java.lang=ALL-UNNAMED", DAON="LATEST")
    print("KoalaNLP 초기화 완료")

    try:
        # 금융 용어 JSON 로드
        financial_terms_path = "./data/financial_terms.json"
        not_verbs_path = "./data/not_verb.json"
        yes_verbs_path = "./data/yes_verb.json"

        financial_terms = load_json_list(financial_terms_path, "terms")
        not_verbs = load_json_list(not_verbs_path, "not_verbs")
        yes_verbs = load_json_list(yes_verbs_path, "yes_verbs")

        print(f"금융 용어 리스트 load 성공")
        print(f"제외 동사 리스트 load 성공")
        print(f"포함 동사 리스트 load 성공")

        # 데이터 디렉토리에서 모든 .txt 파일 처리
        data_dir = "./data/article/"
        article_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]

        if not article_files:
            print("분석할 파일이 없습니다.")
            return

        results = []

        for file_path in article_files:
            print(f"\n=== {os.path.basename(file_path)} 분석 시작 ===")
            file_results = process_file(file_path, financial_terms, not_verbs, yes_verbs)
            results.extend(file_results)

        # 결과 출력
        for result in results:
            print("--------------------------------------------------")
            print(f"기사: {result['article']}")
            print(f"문장: {result['sentence']}")
            print(f"금융 용어: {result['terms']}")
            print(f"동사(관계): {result['verbs']}")
            print("--------------------------------------------------")

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        # KoalaNLP 종료
        finalize()
        print("JVM(KoalaNLP) 종료 완료")


if __name__ == "__main__":
    main()
