# give.py
import json

def load_dictionary(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# 띄어쓰기 제거 및 정의 검색
def search_definition(word, data):
    # 띄어쓰기를 제거한 형태
    word_cleaned = word.replace(" ", "").lower()

    # 이미 사전에 있는 경우
    if word_cleaned in data:
        return data[word_cleaned]

    for key in data.keys():
        # 괄호 앞의 용어 확인
        key_cleaned = key.split("(")[0].replace(" ", "").lower()

        if key_cleaned == word_cleaned:
            return data[key]

        # 괄호 안의 용어 확인
        if "(" in key:
            inside_parentheses = key.split("(")[1].rstrip(")").lower()
            for item in inside_parentheses.split(","):
                if item.strip() == word_cleaned:
                    return data[key]

    return None


def get_word_definition(words, dictionary_file_path):
    if not words:
        return {}

    # 리스트가 아닌 경우 리스트로 변환
    if isinstance(words, str):
        words = [words]

    # 결과를 저장할 딕셔너리
    definitions = {}

    # 기존 사전 로드
    dictionary = load_dictionary(dictionary_file_path)

    # 각 단어에 대해 정의 검색
    for word in words:
        # 사전에 없는 경우, 띄어쓰기를 제거하고 사전에서 검색
        print(f"'{word}'에 대한 정의를 사전에서 검색 중...")
        found_definition = search_definition(word, dictionary)

        if found_definition:
            print(f"{word}: 사전에서 정의를 찾았습니다.")
            definitions[word] = found_definition
        else:
            print(f"'{word}': 정의를 찾을 수 없습니다")

    return definitions
