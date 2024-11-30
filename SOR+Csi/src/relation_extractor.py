#문장에서 동사를 추출하고 yes_verbs에 포함된 동사를 추가로 고려하여 반환
from koalanlp.proc import Tagger
from koalanlp import API


def extract_verbs(sentence, not_verbs, yes_verbs):
    tagger = Tagger(API.DAON)
    analyzed = tagger(sentence)
    verbs = []

    for sent in analyzed:
        for word in sent:
            if hasattr(word, 'morphemes'):  # 형태소와 품사 확인
                for morpheme in word.morphemes:
                    if morpheme.tag.startswith("VV"):  # 동사 태그 확인
                        # 동사의 어간(표면형) + 어미 결합
                        surface = morpheme.surface  # 어간
                        ending = next(
                            (m.surface for m in word.morphemes if m.tag.startswith("EP") or m.tag.startswith("EC") or m.tag.startswith("EF")), ""
                        )  # 어미
                        combined = surface + ending
                        # 동사가 제외 목록에 없거나 yes_verbs에 포함되면 추가
                        if combined not in not_verbs or combined in yes_verbs:
                            verbs.append(combined)

    return list(set(verbs))  # 중복 제거 후 반환
