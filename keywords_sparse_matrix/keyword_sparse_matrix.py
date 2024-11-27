from final.keyword_extractor import KeywordExtractor, KeywordExtractionConfig
from typing import Dict, List
from pathlib import Path
import json
import logging
import numpy as np
from dataclasses import dataclass

# JaccardAnalyzer 클래스 통합
class JaccardAnalyzer:
    def __init__(self, min_word_length: int = 2):
        self.min_word_length = min_word_length

    def create_jaccard_matrix(self, text: str, keywords: List[str]) -> (np.ndarray, List[str]):
        # Implement Jaccard similarity matrix creation
        num_keywords = len(keywords)
        matrix = np.zeros((num_keywords, num_keywords))
        for i in range(num_keywords):
            for j in range(num_keywords):
                if i != j:
                    set1, set2 = set(keywords[i]), set(keywords[j])
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    matrix[i, j] = intersection / union if union > 0 else 0
        return matrix, keywords

    def save_matrix_csv(self, matrix: np.ndarray, keywords: List[str], output_path: str):
        # Save matrix as CSV
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("," + ",".join(keywords) + "\n")
            for i, row in enumerate(matrix):
                f.write(keywords[i] + "," + ",".join(map(str, row)) + "\n")


@dataclass
class AnalysisConfig:
    """Configuration for integrated analysis"""
    keyword_config: KeywordExtractionConfig
    min_word_length: int = 2
    output_dir: str = "analysis_output"
    save_interim: bool = True


class IntegratedAnalyzer:
    """Integrates keyword extraction and Jaccard similarity analysis"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.keyword_extractor = KeywordExtractor(
            article_path="",  # Set dynamically per article
            dictionary_path="econ_dictionary.json",
            config=config.keyword_config
        )
        self.jaccard_analyzer = JaccardAnalyzer(min_word_length=config.min_word_length)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_article(self, article_path: str) -> Dict[str, List[str]]:
        self.keyword_extractor.article_path = article_path
        return self.keyword_extractor.extract_keywords()

    def create_article_matrix(self, article_text: str, keywords: List[str]) -> np.ndarray:
        return self.jaccard_analyzer.create_jaccard_matrix(article_text, keywords)

    def analyze_articles(self, articles_dir: str) -> Dict[str, np.ndarray]:
        articles_path = Path(articles_dir)
        all_matrices = {}
        
        if not articles_path.exists():
            logging.error(f"Articles directory does not exist: {articles_dir}")
            return all_matrices
        
        for article_file in articles_path.glob("*.html"):
            try:
                logging.info(f"Processing article: {article_file}")
                # Extract keywords
                keywords_dict = self.process_article(str(article_file))
                all_keywords = keywords_dict.get('all_keywords', [])
                if not all_keywords:
                    logging.warning(f"No keywords extracted for {article_file}")
                    continue
                
                # Get article text
                with open(article_file, 'r', encoding='utf-8') as f:
                    article_text = f.read()
                
                # Create matrix
                matrix, keywords = self.create_article_matrix(article_text, all_keywords)
                
                # Save individual matrix if configured
                if self.config.save_interim:
                    output_path = self.output_dir / f"matrix_{article_file.stem}.csv"
                    self.jaccard_analyzer.save_matrix_csv(matrix, keywords, str(output_path))
                
                all_matrices[article_file.stem] = matrix
                logging.info(f"Processed article: {article_file.name}")
                
                # Save keywords
                keywords_path = self.output_dir / f"keywords_{article_file.stem}.json"
                with open(keywords_path, 'w', encoding='utf-8') as f:
                    json.dump(keywords_dict, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                logging.error(f"Error processing {article_file}: {e}")
                continue
        
        # Save combined matrices
        if all_matrices:
            combined_path = self.output_dir / "all_matrices.npz"
            np.savez(str(combined_path), **all_matrices)
            logging.info(f"Saved combined matrices to {combined_path}")
        else:
            logging.warning("No matrices were created. Check the input files and extraction logic.")
        
        return all_matrices

        
        # Save combined matrices
        if all_matrices:
            combined_path = self.output_dir / "all_matrices.npz"
            np.savez(str(combined_path), **all_matrices)
            logging.info(f"Saved combined matrices to {combined_path}")
        
        return all_matrices


def main():
    config = AnalysisConfig(
        keyword_config=KeywordExtractionConfig(
            top_n=15,
            similarity_threshold=0.6,
            min_keyword_length=2,
            batch_size=128,
            cache_embeddings=True
        ),
        min_word_length=2,
        output_dir="financial_analysis_output",
        save_interim=True
    )

    analyzer = IntegratedAnalyzer(config)

    try:
        matrices = analyzer.analyze_articles("articles")
        logging.info("Analysis completed successfully")
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
