# src/relation_extractor.py
from koalanlp.proc import Tagger, Dictionary
from koalanlp import API
from koalanlp.Util import initialize, finalize
from koalanlp.types import POS
import json

# JVM 초기화 상태를 확인하는 플래그
jvm_initialized = False

def initialize_jvm():
    global jvm_initialized
    if not jvm_initialized:
        initialize(java_options="-Xmx4g --add-opens java.base/java.util=ALL-UNNAMED --add-opens java.base/java.lang=ALL-UNNAMED", KKMA="LATEST")
        jvm_initialized = True

# KoalaNLP 초기화
initialize_jvm()

# 사전 초기화
KDict = Dictionary(API.KKMA)  # KKMA 분석기 사용

def load_custom_verbs(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        custom_verbs = data.get("yes_verbs", [])
        for verb in custom_verbs:
            KDict.addUserDictionary((verb, POS.VV))  # 사용자 정의 동사 추가
        print(f"사용자 동사 {len(custom_verbs)}개가 사전에 추가되었습니다.")
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
    except json.JSONDecodeError:
        print(f"JSON 파일 형식이 올바르지 않습니다: {file_path}")

def normalize_verbs(verbs, verb_mapping):
    """
    Normalize verbs based on a provided mapping.
    """
    return [verb_mapping.get(verb, verb) for verb in verbs]

def extract_verbs(sentence, not_verbs, yes_verbs, verb_mapping=None):
    tagger = Tagger(API.KKMA)
    analyzed = tagger(sentence)
    verbs = []

    for sent in analyzed:
        for word in sent:
            if hasattr(word, 'morphemes'):  # 형태소 확인
                base_form = None
                endings = []

                for morpheme in word.morphemes:
                    # 동사 어간 확인 (VV, VX)
                    if morpheme.tag.startswith("VV") or morpheme.tag.startswith("VX"):
                        base_form = morpheme.surface
                    # 어미 확인 (EP, EC, EF)
                    elif morpheme.tag.startswith("EP") or morpheme.tag.startswith("EC") or morpheme.tag.startswith("EF"):
                        endings.append(morpheme.surface)

                if base_form:
                    combined = base_form + "".join(endings)
                    if combined not in not_verbs or combined in yes_verbs:
                        verbs.append(combined)

    # 사용자 정의 사전에서 동사 추가
    custom_verbs = [entry[0] for entry in KDict.getItems() if entry[1] == POS.VV]
    verbs.extend([verb for verb in custom_verbs if verb in sentence])

    unique_verbs = list(set(verbs))
    if verb_mapping:
        return normalize_verbs(unique_verbs, verb_mapping)
    return unique_verbs

# KoalaNLP 종료
def cleanup():
    finalize()
