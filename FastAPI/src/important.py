# important.py
import json
from collections import Counter


def filter_top_5_verbs_relationships_with_limited_pairs(json_file_path: str, output_file_path: str, top_n: int = 5, max_pairs_per_verb: int = 3):
    """
    상위 N개의 빈도 높은 동사에 대한 관계를 필터링하고, 동사당 최대 키워드 쌍의 수를 제한하여 저장합니다.

    Args:
        json_file_path (str): 입력 JSON 파일 경로.
        output_file_path (str): 필터링된 JSON 파일을 저장할 경로.
        top_n (int): 분석 및 포함할 상위 동사 개수.
        max_pairs_per_verb (int): 동사당 저장할 최대 관계 수.
    """
    with open(json_file_path, 'r', encoding='utf-8') as file:
        # 파일명 추가를 위해 데이터가 파일명별로 구조화되었는지 확인
        data = json.load(file)

    # 파일명 추출 (파일 구조에 따라 다름)
    if isinstance(data, dict):
        file_name, entries = next(iter(data.items()))  # 첫 번째 파일명과 데이터 추출
    else:
        raise ValueError("JSON 데이터는 파일명을 포함한 딕셔너리 형식이어야 합니다.")

    # 각 동사의 빈도 계산
    verb_counter = Counter(entry["verb"] for entry in entries)
    most_common_verbs = verb_counter.most_common(top_n)

    # 상위 N개의 동사 출력
    print(f"상위 {top_n}개의 동사:")
    for verb, count in most_common_verbs:
        print(f"동사: {verb}, 빈도: {count}")

    # 상위 N개의 동사에 해당하는 관계만 필터링
    top_verbs = {verb for verb, _ in most_common_verbs}
    filtered_data = [
        entry for entry in entries
        if entry["verb"] in top_verbs
    ]

    # 동사별 키워드 쌍 구성 및 제한
    limited_relationships = []
    verb_to_pairs = {verb: [] for verb in top_verbs}

    for entry in filtered_data:
        verb = entry["verb"]
        keywords = tuple(sorted(entry["keywords"]))  # 키워드를 정렬하여 일관성 유지
        verb_to_pairs[verb].append(keywords)

    # 키워드 쌍을 빈도수로 정렬하고 최대 관계 수로 제한
    for verb, keyword_pairs in verb_to_pairs.items():
        # 키워드 쌍 빈도 계산
        pair_counter = Counter(keyword_pairs)
        top_pairs = pair_counter.most_common(max_pairs_per_verb)
        for keywords, _ in top_pairs:
            limited_relationships.append({
                "verb": verb,
                "keywords": list(keywords)
            })

    # 최종 데이터에 파일명 포함
    final_data = {file_name: limited_relationships}

    # 제한된 관계를 출력 파일에 저장
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(final_data, file, ensure_ascii=False, indent=4)

    print(f"\n상위 {top_n}개의 동사와 동사당 최대 {max_pairs_per_verb}개의 관계가 {output_file_path}에 저장되었습니다.")
