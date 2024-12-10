# src/sentence_filter.py
# 기사 이해를 위한 경제 금융 용어 개수가 min_count 이상 max_count 이하인 문장만 반환
def filter_sentences_by_term_count(sentence_with_terms, min_count=2, max_count=2):
    return [
        (sentence, terms)
        for sentence, terms in sentence_with_terms
        if min_count <= len(terms) <= max_count
    ]
