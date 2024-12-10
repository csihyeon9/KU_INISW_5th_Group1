# give.py
import requests
from bs4 import BeautifulSoup
import json


def load_dictionary(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


def save_dictionary(file_path, dictionary):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(dictionary, file, ensure_ascii=False, indent=4)


def fetch_definition_hankyung(word):
    url = f"https://dic.hankyung.com/economy/list?word={word}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        try:
            contents_div = soup.find("div", {"id": "contents", "class": "contents"})
            if contents_div:
                results = contents_div.find_all("li")
                for result in results:
                    title = result.find("h4").get_text(strip=True).replace(" ", "")
                    if word == title:
                        definition = result.find("p").text.strip()
                        return definition
                return None
            else:
                return "'{word}': <div id='contents'>를 찾을 수 없습니다."
        except Exception as e:
            print(f"'{word}' 처리 중 오류 발생: {e}")
            return None
    else:
        print(f"'{word}' 요청 실패: 상태 코드 {response.status_code}")
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
        # 이미 사전에 있는 경우
        if word in dictionary:
            print(f"{word}: 사전에서 정의를 찾았습니다.")
            definitions[word] = dictionary[word]
            continue

        # 사전에 없는 경우 한경 사전에서 검색
        print(f"'{word}'에 대한 정의를 한경 사전에서 검색 중...")
        definition = fetch_definition_hankyung(word)

        if definition:
            print(f"{word}: 정의를 찾았습니다")
            definitions[word] = definition
            # 새로운 정의를 사전에 추가
            dictionary[word] = definition
        else:
            print(f"'{word}': 정의를 찾을 수 없습니다")
            definitions[word] = None

    # 업데이트된 사전 저장
    save_dictionary(dictionary_file_path, dictionary)

    return definitions