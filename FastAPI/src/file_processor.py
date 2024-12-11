# file_processor.py
# 필요한 라이브러리 임포트
from koalanlp.proc import Tagger
from koalanlp import API
from koalanlp.Util import initialize, finalize
import kss
from typing import Any, List, Tuple

# JVM 초기화 상태 플래그
jvm_initialized = False

# KoalaNLP를 위한 JVM 초기화
def initialize_jvm():
    """
    JVM이 초기화되지 않은 경우 초기화합니다.
    KoalaNLP의 DAON API를 사용할 수 있도록 설정합니다.
    """
    global jvm_initialized
    if not jvm_initialized:
        initialize(java_options="-Xmx4g --add-opens java.base/java.util=ALL-UNNAMED --add-opens java.base/java.lang=ALL-UNNAMED", DAON="LATEST")
        jvm_initialized = True

# KoalaNLP 초기화
initialize_jvm()

# --- 파일 로더 ---
def load_article(file_path):
    """
    텍스트 파일의 내용을 문자열로 로드합니다.

    Args:
        file_path (str): 텍스트 파일 경로.

    Returns:
        str: 파일 내용.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# --- 문장 분리 ---
def split_sentences(document):
    """
    문서를 문장 리스트로 분리합니다. KSS를 사용합니다.

    Args:
        document (str): 입력 문서 텍스트.

    Returns:
        List[str]: 분리된 문장들의 리스트.
    """
    return list(kss.split_sentences(document))

# --- 동사와 명사 추출 ---
def extract_verbs_and_nouns(sentence: Any) -> Tuple[List[str], List[str]]:
    """
    DAON을 사용하여 문장에서 특정 동사(VV, VX)와 명사(NNG, NNP, NNBC)를 추출합니다.

    Args:
        sentence (Any): 처리할 문장. KoalaNLP와 호환 가능한 형식.

    Returns:
        Tuple[List[str], List[str]]: 두 개의 리스트를 포함하는 튜플 - 동사(VV, VX)와 명사(NNG, NNP, NNBC).
    """
    tagger = Tagger(API.DAON)  # DAON 태거 초기화
    analyzed = tagger(sentence)
    verbs = []
    nouns = []

    for sent in analyzed:
        for word in sent:
            if hasattr(word, 'morphemes'):  # 형태소 속성이 있는 경우
                base_form = None
                endings = []

                for morpheme in word.morphemes:
                    # VV 또는 VX로 태깅된 동사만 포함
                    if morpheme.tag in {"VV", "VX"}:
                        base_form = morpheme.surface
                    elif morpheme.tag.startswith("EP") or morpheme.tag.startswith("EC") or morpheme.tag.startswith("EF"):
                        endings.append(morpheme.surface)

                    # NNG, NNP, NNBC로 태깅된 명사만 포함
                    elif morpheme.tag in {"NNG", "NNP", "NNBC"}:
                        nouns.append(morpheme.surface)

                # 동사의 기본형과 어미를 결합하여 완성된 동사 생성
                if base_form:
                    combined_verb = base_form + "".join(endings)
                    verbs.append(combined_verb)

    # 중복을 제거한 뒤 결과 반환
    return list(set(verbs)), list(set(nouns))

# KoalaNLP 정리 함수
def cleanup():
    """
    JVM을 종료하고 KoalaNLP 자원을 해제합니다.
    """
    finalize()
