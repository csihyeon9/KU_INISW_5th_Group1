# src/term_extractor.py
#문장에서 금융 용어만 추출하여 리스트로 반환
def extract_terms(sentence, article_terms):
    return [term for term in article_terms if term in sentence]
