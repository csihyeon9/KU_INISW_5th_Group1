import json
import math
from collections import defaultdict
from tqdm import tqdm

class PMIProcessor:
    def __init__(self, data_path, unique_keywords_path):
        self.data_path = data_path
        self.unique_keywords_path = unique_keywords_path
        self.categories = {
            "1": "인과or상관성",
            "2": "변화&추세",
            "3": "시장및거래관계",
            "4": "정책&제도",
            "5": "기업활동",
            "6": "금융상품&자산",
            "7": "위험및위기",
            "8": "기술및혁신"
        } # 카테고리 매핑
        self.unique_keywords = set()  # unique_keywords 저장
        self.keyword_counts = defaultdict(lambda: defaultdict(int))  # 카테고리별 키워드 등장 횟수
        
        self.pair_counts = defaultdict(lambda: defaultdict(int))  # 카테고리별 키워드 쌍 등장 횟수
        self.category_counts = defaultdict(int)  # 카테고리별 전체 관계 수
        
        self.total_relations = 0  # 전체 관계 수

    def load_unique_keywords(self):
        """Load unique keywords from file"""
        with open(self.unique_keywords_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.unique_keywords = set(data["unique_keywords"])
        print(f"Loaded {len(self.unique_keywords)} unique keywords.")

    def load_data(self):
        """ 
        데이터셋에서 각 카테고리별 키워드 및 키워드 쌍의 등장 횟수를 계산.
        self.keyword_counts: 카테고리별 키워드 등장 횟수 저장.
        self.pair_counts: 카테고리별 키워드 쌍 등장 횟수 저장.
        self.category_counts: 카테고리별 전체 관계 수 저장.
        
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for article in tqdm(data, desc="Processing articles"):
            relations = article.get("relations", {})
            for category, relation_list in relations.items():
                if category not in self.categories:
                    continue
                
                self.category_counts[category] += len(relation_list)
                self.total_relations += len(relation_list)

                for relation in relation_list:
                    # Filter keywords based on unique_keywords
                    keywords = [
                        kw for kw in relation[1:] if kw in self.unique_keywords
                    ]
                    for keyword in keywords:
                        self.keyword_counts[category][keyword] += 1
                    
                    if len(keywords) == 2:
                        # 키워드 쌍 등장 횟수 추가
                        sorted_pair = tuple(sorted(keywords))
                        self.pair_counts[category][sorted_pair] += 1

    def calculate_pmi(self):
        """카테고리별 키워드 쌍의 PMI를 계산"""
        pmi_values = defaultdict(dict)

        for category, pairs in self.pair_counts.items():
            total_category_relations = self.category_counts[category]
            
            for pair, pair_count in pairs.items():
                w1, w2 = pair
                p_w1_w2_c = pair_count / total_category_relations
                p_w1_c = self.keyword_counts[category][w1] / total_category_relations
                p_w2_c = self.keyword_counts[category][w2] / total_category_relations

                if p_w1_c > 0 and p_w2_c > 0:
                    pmi = math.log(p_w1_w2_c / (p_w1_c * p_w2_c))
                    pmi_values[category][pair] = pmi
                else:
                    pmi_values[category][pair] = 0  # PMI가 정의되지 않는 경우
        
        return pmi_values

    def save_pmi(self, output_path, pmi_values):
        """
        결과를 JSON 파일로 저장.
        각 카테고리의 이름과 키워드 쌍을 사람이 읽기 쉽게 변환
        """
        serializable_pmi = {
            self.categories[category]: {
                f"{pair[0]} | {pair[1]}": value
                for pair, value in pairs.items()
            }
            for category, pairs in pmi_values.items()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_pmi, f, ensure_ascii=False, indent=2)
        print(f"PMI values saved to {output_path}")

def main():
    data_path = "data/dataset/final_dataset3.json"
    unique_keywords_path = "data/updated_unique_keywords.json"
    output_path = "data/pairwise_pmi_values3.json"

    processor = PMIProcessor(data_path, unique_keywords_path)
    processor.load_unique_keywords()
    processor.load_data()
    
    pmi_values = processor.calculate_pmi()
    processor.save_pmi(output_path, pmi_values)

if __name__ == "__main__":
    main()
