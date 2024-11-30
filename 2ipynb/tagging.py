import pandas as pd
import nltk
from nltk import Tree
from konlpy.tag import *

hannanum = Hannanum()
kkma = Kkma()
komoran = Komoran()
okt = Okt()

df3 = pd.read_csv('result.csv',encoding='cp949')
df3.head()


data10 = sentences[0].split()

sent = []
sub = []
obj = []
rel = []

for sentence in data10 :
    tokens = komoran.pos(sentence)
    sent.append(sentence)
    for word, pos in tokens :
        if pos == 'JKO'or pos =='JKS' :
           if pos =='JKS' :
               sub.append(' '.join(sent))   
               sent = []
               
           elif pos =='JKO' :
               obj.append(' '.join(sent))
               sent = [] 
        if pos =='XSV' :
            rel.append(' '.join(sent))
            sent = []
        if pos =='VV' :
            rel.append(' '.join(sent))
            sent = []     

print(sub)
print(obj)
print(rel)


sub_text = sub[-1].split()
obj_text = obj[-1].split()
rel_text = rel[-1].split()
print(sub_text)
print(obj_text)
print(rel_text)


subject = []
object = []
relation =[]

for sentence in sub_text :
    tokens = komoran.pos(sentence)
    pos_tags = [pos for word, pos in tokens]
    if 'JKS' not in pos_tags :
        subject.append(sentence)
    else :
        word_list = [word for word, pos in tokens if pos != 'JKS']
        joined_word = ''.join(word_list)
        subject.append(joined_word)
        
for sentence in obj_text :
    tokens = komoran.pos(sentence)
    pos_tags = [pos for word, pos in tokens]
    if 'JKO' not in pos_tags :
        continue
    else :
        word_list = [word for word, pos in tokens if pos != 'JKO']
        joined_word = ''.join(word_list)
        object.append(joined_word)
        
for sentence in rel_text :
    tokens = komoran.pos(sentence)
    pos_tags = [pos for word, pos in tokens]
    if 'XSV' not in pos_tags  :
        continue
    else :
        word_list = [word for word, pos in tokens if pos == 'NNG']
        joined_word = ''.join(word_list)
        relation.append(joined_word)
        
for sentence in rel_text :
    tokens = komoran.pos(sentence)
    pos_tags = [pos for word, pos in tokens]
    if 'VV' not in pos_tags :
        continue
    else :
        relation.append(sentence)
        
        
print(subject)
print(object)
print(relation)



data10 = sentences[0].split()

sent = []
sub = []
obj = []
rel = []

for sentence in data10 :
    tokens = komoran.pos(sentence)
    sent.append(sentence)
    for word, pos in tokens :
        if pos == 'JKO'or pos =='JKS' :
           if pos =='JKS' :
               sub.append(' '.join(sent))   
               sent = []
               
           elif pos =='JKO' :
               obj.append(' '.join(sent))
               sent = [] 
        if pos =='XSV' :
            rel.append(' '.join(sent))
            sent = []
        if pos =='VV' :
            rel.append(' '.join(sent))
            sent = []    
            
sub_text = sub[-1].split()
obj_text = obj[-1].split()
rel_text = rel[-1].split()           
            
subject = []
object = []
relation =[]

for sentence in sub_text :
    tokens = komoran.pos(sentence)
    pos_tags = [pos for word, pos in tokens]
    if 'JKS' not in pos_tags :
        subject.append(sentence)
    else :
        word_list = [word for word, pos in tokens if pos != 'JKS']
        joined_word = ''.join(word_list)
        subject.append(joined_word)
        
for sentence in obj_text :
    tokens = komoran.pos(sentence)
    pos_tags = [pos for word, pos in tokens]
    if 'JKO' not in pos_tags :
        continue
    else :
        word_list = [word for word, pos in tokens if pos != 'JKO']
        joined_word = ''.join(word_list)
        object.append(joined_word)
        
for sentence in rel_text :
    tokens = komoran.pos(sentence)
    pos_tags = [pos for word, pos in tokens]
    if 'XSV' not in pos_tags  :
        continue
    else :
        word_list = [word for word, pos in tokens if pos == 'NNG']
        joined_word = ''.join(word_list)
        relation.append(joined_word)
        
for sentence in rel_text :
    tokens = komoran.pos(sentence)
    pos_tags = [pos for word, pos in tokens]
    if 'VV' not in pos_tags :
        continue
    else :
        relation.append(sentence)


        
print("subject :",subject)
print("object :",object)
print("relation :",relation)
print(sentences[0])


results = []  # 데이터를 임시 저장할 리스트

for df_sent in sentences:
    data10 = df_sent.split()
    
    sent = []
    sub = []
    obj = []
    rel = []

    for sentence in data10:
        tokens = komoran.pos(sentence)
        sent.append(sentence)
        for word, pos in tokens:
            if pos == 'JKO' or pos == 'JKS':
                if pos == 'JKS':
                    sub.append(' '.join(sent))
                    sent = []
                elif pos == 'JKO':
                    obj.append(' '.join(sent))
                    sent = [] 
            if pos == 'XSV':
                rel.append(' '.join(sent))
                sent = []
            if pos == 'VV':
                rel.append(' '.join(sent))
                sent = []    

    # 리스트가 비어있으면 기본값 설정
    sub_text = sub[-1].split() if sub else []
    obj_text = obj[-1].split() if obj else []
    rel_text = rel[-1].split() if rel else []

    subject = []
    object = []
    relation = []

    for sentence in sub_text:
        tokens = komoran.pos(sentence)
        pos_tags = [pos for word, pos in tokens]
        if 'JKS' not in pos_tags:
            subject.append(sentence)
        else:
            word_list = [word for word, pos in tokens if pos != 'JKS']
            joined_word = ''.join(word_list)
            subject.append(joined_word)

    for sentence in obj_text:
        tokens = komoran.pos(sentence)
        pos_tags = [pos for word, pos in tokens]
        if 'JKO' not in pos_tags:
            continue
        else:
            word_list = [word for word, pos in tokens if pos != 'JKO']
            joined_word = ''.join(word_list)
            object.append(joined_word)

    for sentence in rel_text:
        tokens = komoran.pos(sentence)
        pos_tags = [pos for word, pos in tokens]
        if 'XSV' not in pos_tags:
            continue
        else:
            word_list = [word for word, pos in tokens if pos == 'NNG']
            joined_word = ''.join(word_list)
            relation.append(joined_word)

    for sentence in rel_text:
        tokens = komoran.pos(sentence)
        pos_tags = [pos for word, pos in tokens]
        if 'VV' not in pos_tags:
            continue
        else:
            relation.append(sentence)

    subject = ' '.join(subject)
    object = ' '.join(object)
    relation = ' '.join(relation)
    
    results.append({
        'text': df_sent,
        'subject': subject,
        'object': object,
        'relation': relation
    })

# 모든 데이터를 한 번에 DataFrame으로 변환
df = pd.DataFrame(results)

import pandas as pd

df = pd.DataFrame(columns=['text', 'subject', 'object', 'relation'])

results = []  # 데이터를 임시로 저장할 리스트

for df_sent in sentences:
    data10 = df_sent.split()
    sent = []
    sub = []
    obj = []
    rel = []

    for sentence in data10:
        tokens = komoran.pos(sentence)
        sent.append(sentence)
        for word, pos in tokens:
            if pos == 'JKO' or pos == 'JKS':
                if pos == 'JKS':
                    sub.append(' '.join(sent))
                    sent = []
                elif pos == 'JKO':
                    obj.append(' '.join(sent))
                    sent = []
            if pos == 'XSV' or pos == 'VV':
                rel.append(' '.join(sent))
                sent = []

    # 비어 있는 경우 기본값으로 설정
    sub_text = sub[-1].split() if sub else []
    obj_text = obj[-1].split() if obj else []
    rel_text = rel[-1].split() if rel else []

    subject = []
    object = []
    relation = []

    for sentence in sub_text:
        tokens = komoran.pos(sentence)
        pos_tags = [pos for word, pos in tokens]
        if 'JKS' not in pos_tags:
            subject.append(sentence)
        else:
            word_list = [word for word, pos in tokens if pos != 'JKS']
            joined_word = ''.join(word_list)
            subject.append(joined_word)

    for sentence in obj_text:
        tokens = komoran.pos(sentence)
        pos_tags = [pos for word, pos in tokens]
        if 'JKO' not in pos_tags:
            continue
        else:
            word_list = [word for word, pos in tokens if pos != 'JKO']
            joined_word = ''.join(word_list)
            object.append(joined_word)

    for sentence in rel_text:
        tokens = komoran.pos(sentence)
        pos_tags = [pos for word, pos in tokens]
        if 'XSV' in pos_tags:
            word_list = [word for word, pos in tokens if pos == 'NNG']
            joined_word = ''.join(word_list)
            relation.append(joined_word)
        elif 'VV' in pos_tags:
            relation.append(sentence)

    subject = ' '.join(subject)
    object = ' '.join(object)
    relation = ' '.join(relation)

    # 결과 저장
    results.append({
        'text': df_sent,
        'subject': subject,
        'object': object,
        'relation': relation
    })

# 한 번에 DataFrame 생성
df = pd.DataFrame(results)
print(df)

df.to_csv('new_data.csv',encoding='cp949')