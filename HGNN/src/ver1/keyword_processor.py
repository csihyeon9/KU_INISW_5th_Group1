import json
import math
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm

class KeywordAndVerbProcessor:
    def __init__(self, unique_keywords_path: str, news_data_path: str):
        self.unique_keywords_path = unique_keywords_path
        self.news_data_path = news_data_path
        self.keyword2idx = {}
        self.verb_categories = {
            "1": "인과or상관성",
            "2": "변화&추세",
            "3": "시장및거래관계",
            "4": "정책&제도",
            "5": "기업활동",
            "6": "금융상품&자산",
            "7": "위험및위기",
            "8": "기술및혁신",
        }
        self.pair_counts = defaultdict(int)
        self.category_counts = defaultdict(int)
        self.cooccurrence_counts = defaultdict(lambda: defaultdict(int))
        self.verb_single_keyword_counts = defaultdict(lambda: defaultdict(int))
        self.total_count = 0

    def load_data(self):
        with open(self.unique_keywords_path, 'r', encoding='utf-8') as f:
            unique_keywords_data = json.load(f)
            unique_keywords = set(unique_keywords_data["unique_keywords"])
        
        with open(self.news_data_path, 'r', encoding='utf-8') as f:
            news_data = json.load(f)

        for article in tqdm(news_data, desc="Processing articles"):
            relations = article.get("relations", {})
            if not isinstance(relations, dict):
                print(f"Error in article: {article}")
                continue

            article_keywords = set()
            for category, relation_list in relations.items():
                mapped_category = self.verb_categories.get(category, "Unknown")
                self.category_counts[mapped_category] += len(relation_list)

                for relation in relation_list:
                    keywords_in_relation = self.extract_keywords_from_relation(relation)
                    article_keywords.update(keywords_in_relation)

                    if len(keywords_in_relation) == 2:
                        sorted_pair = tuple(sorted(keywords_in_relation))
                        self.pair_counts[sorted_pair] += 1
                        self.cooccurrence_counts[mapped_category][sorted_pair] += 1
                    
                    for keyword in keywords_in_relation:
                        if keyword in unique_keywords:
                            self.verb_single_keyword_counts[mapped_category][(relation[0], keyword)] += 1

            self.total_count += len(article_keywords)

    def extract_keywords_from_relation(self, relation):
        if len(relation) == 3:
            if relation[0].endswith(("치다", "되다")) or relation[2].endswith("하다"):
                return [relation[1], relation[2]]
            elif relation[2].endswith(("치다", "되다", "하다")):
                return [relation[0], relation[1]]
        elif len(relation) == 2:
            return [relation[1]]
        return []

    def calculate_pmi(self):
        pmi_values = defaultdict(dict)
        for category, pairs in self.cooccurrence_counts.items():
            for pair, count in pairs.items():
                p_w1_w2_c = count / self.total_count
                p_w1_w2 = self.pair_counts[pair] / self.total_count
                p_c = self.category_counts[category] / self.total_count

                if p_w1_w2 > 0 and p_c > 0:
                    pmi = math.log(p_w1_w2_c / (p_w1_w2 * p_c))
                    pmi_values[category][pair] = pmi
        
        return pmi_values

    def calculate_verb_single_keyword_pmi(self):
        verb_single_pmi_values = defaultdict(dict)
        for category, verb_keywords in self.verb_single_keyword_counts.items():
            for (verb, keyword), count in verb_keywords.items():
                p_v_k_c = count / self.total_count
                p_v_k = count / self.total_count
                p_c = self.category_counts[category] / self.total_count

                if p_v_k > 0 and p_c > 0:
                    pmi = math.log(p_v_k_c / (p_v_k * p_c))
                    verb_single_pmi_values[category][(verb, keyword)] = pmi
        
        return verb_single_pmi_values

    def save_pmi(self, output_path: str, pmi_values):
        serializable_pmi_values = {}
        for category, pairs in pmi_values.items():
            serializable_pmi_values[category] = {
                f"{pair[0]} | {pair[1]}": value 
                for pair, value in pairs.items()
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_pmi_values, f, ensure_ascii=False, indent=2)
        print(f"PMI values saved to {output_path}")

def main():
    unique_keywords_path = "data/unique_keywords.json"
    news_data_path = "data/final_dataset2.json"
    
    pairwise_pmi_path = "data/pairwise_pmi_values.json"
    verb_single_keyword_pmi_path = "data/verb_single_keyword_pmi_values.json"

    processor = KeywordAndVerbProcessor(unique_keywords_path, news_data_path)
    processor.load_data()
    
    pairwise_pmi = processor.calculate_pmi()
    processor.save_pmi(pairwise_pmi_path, pairwise_pmi)
    
    verb_single_keyword_pmi = processor.calculate_verb_single_keyword_pmi()
    processor.save_pmi(verb_single_keyword_pmi_path, verb_single_keyword_pmi)

if __name__ == "__main__":
    main()