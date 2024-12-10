import requests
import json
import time

# AJAX 요청을 위한 기본 URL
url = 'https://dic.hankyung.com/work/economyPhonemeList'
words = ['ㅇ']
terms_dict = {}


def fetch_terms(word, page):
    for attempt in range(10):  # 최대 5회 재시도
        try:
            response = requests.post(url, data={'word': word, 'page': page})
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"요청 실패: {e}, {attempt + 1}회 재시도 중...")
            time.sleep(5)  # 5초 후 재시도
    return None


# 모든 글자에 대해 용어 가져오기
for word in words:
    page = 0
    while True:
        page += 1
        terms_data = fetch_terms(word, page)
        if terms_data and len(terms_data) > 0:
            for term in terms_data:
                name = term['WORD']
                definition = term['CONTENT1']

                # 영어 표현 및 한자 가져오기
                expressions = []
                if 'CHINESE' in term and term['CHINESE']:
                    expressions.append(term['CHINESE'])
                if 'EWORD1' in term and term['EWORD1']:
                    expressions.append(term['EWORD1'])
                if 'EWORD2' in term and term['EWORD2']:
                    expressions.append(term['EWORD2'])
                if 'SIMPLE' in term and term['SIMPLE']:
                    expressions.append(term['SIMPLE'])

                # 표현이 있을 경우() 형식으로 추가
                if expressions:
                    expressions_part = "({})".format(", ".join(expressions))
                    terms_dict[f"{name}{expressions_part}"] = definition
                else:
                    terms_dict[name] = definition
        else:
            print(f"{word}에 대한 용어가 더 이상 없습니다.")
            break

# JSON 파일로 저장
with open('economy_terms.json', 'w', encoding='utf-8') as json_file:
    json.dump(terms_dict, json_file, ensure_ascii=False, indent=4)

print("모든 경제 용어가 'economy_terms.json' 파일에 저장되었습니다.")
