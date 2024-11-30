import json
import nltk
import pandas as pd
from konlpy.tag import *

hannanum = Hannanum()
kkma = Kkma()
komoran = Komoran()
okt = Okt()

nltk.download('punkt') 

file_path = "./data"

with open('./data/news.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

df_list = []

# JSON 데이터를 바로 순회
if '경제' in json_data['metadata']['original_topic']:
    for paragraph in json_data['paragraph']:
        df_list.append(paragraph['form'])

df = pd.DataFrame(df_list)


df_nltk = []
for text in df_list : 
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences :
        df_nltk.append(sentence)
        
df = pd.DataFrame(df_nltk)


clause_list = []
for text in df_list[:3]:
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        clause_list.append(pos_tags)

df = pd.DataFrame(clause_list)
print(df)

df_nltk[:10]


exm = [
    ['문장1', 'POS1'],
    ['문장2', 'POS2'],
    ['문장3', 'POS3']
]

df_pd = pd.DataFrame(exm, columns=['df_txt', 'df_pos'])
a = df_pd['df_pos']

df_J = []

for sentence in df_nltk:
    exm = komoran.pos(sentence)
    df_pd = pd.DataFrame(exm, columns=['df_txt', 'df_pos'])
    df_pos = df_pd['df_pos']
    if 'JKS' in df_pos == True and 'JKO' in df_pos == True:
        df_J.append(sentence)


from konlpy.tag import Komoran

komoran = Komoran()

# 'data'에 샘플 텍스트를 정의합니다.
data = "이것은 샘플 텍스트입니다."

# Komoran 형태소 분석 수행
exm = komoran.pos(data)

# DataFrame 생성
df_pd = pd.DataFrame(exm, columns=['df_txt', 'df_pos'])
df_pd

result = []
for sentence in df_nltk:
    tokens = komoran.pos(sentence)
    pos_tags = [pos for word, pos in tokens]
    if 'JKS' in pos_tags and 'JKO' in pos_tags:
        result.append(sentence)

len(df_nltk)
len(result)


result = pd.DataFrame(result)
result.to_csv('result.csv',encoding='cp949')


df1 = []
df2 = []

for sentence in result:
    if not isinstance(sentence, str):
        continue  # 문자열이 아닌 경우 건너뜁니다.

    tokens = komoran.pos(sentence)
    pos_tags = [pos for word, pos in tokens]

    if pos_tags.count('JKS') == 1 and pos_tags.count('JKO') == 1:
        df2.append(sentence)

df3 = []
for text in df1 :
    if '“' in text :
        continue
    else :
        df3.append(text)
        
len(df3)