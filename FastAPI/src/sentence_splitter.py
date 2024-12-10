# src/sentence_splitter.py
# KSS를 사용하여 문서를 문장 단위로 분리하여 리스트로 반환
import kss

def split_sentences(document):
    return list(kss.split_sentences(document))
