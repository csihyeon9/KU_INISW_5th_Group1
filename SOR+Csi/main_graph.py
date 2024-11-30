import os
import json
from src.file_loader import load_article
from src.sentence_splitter import split_sentences
from src.term_extractor import extract_terms
from src.sentence_filter import filter_sentences_by_term_count
from src.relation_extractor import extract_verbs
from koalanlp.Util import initialize, finalize
from pyvis.network import Network


def load_json_list(json_path, key):
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data.get(key, [])
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {json_path}")
        return []
    except json.JSONDecodeError:
        print(f"JSON 파일 형식이 올바르지 않습니다: {json_path}")
        return []


def process_file(file_path, financial_terms, not_verbs, yes_verbs):
    document = load_article(file_path)
    sentences = split_sentences(document)
    sentence_with_terms = [
        (sentence, extract_terms(sentence, financial_terms)) for sentence in sentences
    ]
    filtered_sentences = filter_sentences_by_term_count(sentence_with_terms, min_count=2, max_count=2)

    results = []

    for sentence, terms in filtered_sentences:
        verbs = extract_verbs(sentence, not_verbs, yes_verbs)
        if verbs:  # 동사가 있는 경우만 처리
            results.append({
                "article": os.path.basename(file_path),
                "sentence": sentence,
                "terms": terms,
                "verbs": verbs
            })

    return results


def create_interactive_graph(results, output_path):
    net = Network(height="1000px", width="100%", directed=True, notebook=False)

    for idx, result in enumerate(results):
        # 문장 노드 생성
        sentence_node = f"문장 {idx + 1}"
        net.add_node(
            sentence_node,
            label="",  # 노드에 표시되는 내용을 빈값으로
            title=result["sentence"],  # 팝업에 전체 문장 표시
            shape="box",
            color="lightblue"
        )

        # 금융 용어와 동사 노드 연결
        term_nodes = []
        for term in result["terms"]:
            term_node = f"{sentence_node}_금융용어_{term}"
            net.add_node(term_node, label=term, shape="circle", color="lightgreen")
            net.add_edge(sentence_node, term_node)
            term_nodes.append(term_node)

        # 동사 관계 연결
        for i, term1 in enumerate(term_nodes):
            for j, term2 in enumerate(term_nodes):
                if i < j:  # 중복 연결 방지
                    for verb in result["verbs"]:
                        edge_label = f"{verb} ({idx + 1})"
                        net.add_edge(term1, term2, label=edge_label, title=edge_label)

    # 그래프 옵션 설정
    net.set_options("""
    var options = {
      "nodes": {
        "font": {"size": 12, "align": "center"},
        "shape": "circle"
      },
      "edges": {
        "smooth": true,
        "arrows": {
          "to": {"enabled": true}
        }
      },
      "physics": {
        "enabled": true
      }
    }
    """)

    # 그래프 저장
    net.show(output_path, notebook=False)


def main():
    # KoalaNLP 초기화
    initialize(java_options="-Xmx4g --add-opens java.base/java.util=ALL-UNNAMED --add-opens java.base/java.lang=ALL-UNNAMED", DAON="LATEST")
    print("KoalaNLP 초기화 완료")

    try:
        # 금융 용어 JSON 로드
        financial_terms_path = "./data/financial_terms.json"
        not_verbs_path = "./data/not_verb.json"
        yes_verbs_path = "./data/yes_verb.json"

        financial_terms = load_json_list(financial_terms_path, "terms")
        not_verbs = load_json_list(not_verbs_path, "not_verbs")
        yes_verbs = load_json_list(yes_verbs_path, "yes_verbs")

        print(f"금융 용어 리스트 load 성공")
        print(f"제외 동사 리스트 load 성공")
        print(f"포함 동사 리스트 load 성공")

        # 데이터 디렉토리에서 모든 .txt 파일 처리
        data_dir = "./data/article/"
        article_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]

        if not article_files:
            print("분석할 파일이 없습니다.")
            return

        all_results = []

        for file_path in article_files:
            print(f"\n=== {os.path.basename(file_path)} 분석 시작 ===")
            file_results = process_file(file_path, financial_terms, not_verbs, yes_verbs)
            all_results.extend(file_results)

        # 결과를 HTML 그래프로 저장
        output_path = "interactive_results_graph.html"
        create_interactive_graph(all_results, output_path)
        print(f"결과 통합 그래프가 {output_path}에 저장되었습니다.")

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        # KoalaNLP 종료
        finalize()
        print("JVM 종료 완료")


if __name__ == "__main__":
    main()
