#문장에서 금융 용어만 추출하여 리스트로 반환
def extract_terms(sentence, financial_terms):
    return [term for term in financial_terms if term in sentence]
