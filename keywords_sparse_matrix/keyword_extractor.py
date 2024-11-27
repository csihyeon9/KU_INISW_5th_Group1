from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Optional, Tuple, Union
from bs4 import BeautifulSoup
from kiwipiepy import Kiwi
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel
from keybert import KeyBERT
from pathlib import Path
import logging
import json
import re
import os
import pickle
import torch
import numpy as np
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KeywordExtractionConfig:
    """Configuration for keyword extraction parameters"""
    top_n: int = 10
    similarity_threshold: float = 0.6
    min_keyword_length: int = 2
    batch_size: int = 128
    cache_embeddings: bool = True

class KeywordExtractor:
    def __init__(
        self,
        article_path: Union[str, Path],
        dictionary_path: Union[str, Path],
        config: Optional[KeywordExtractionConfig] = None
    ):
        """
        Initialize the financial keyword extractor

        Args:
            article_path: Path to article HTML file
            dictionary_path: Path to financial terms dictionary JSON file
            config: Configuration parameters for keyword extraction
        """
        self.article_path = Path(article_path)
        self.dictionary_path = Path(dictionary_path)
        self.config = config or KeywordExtractionConfig()
        self.embedding_cache_path = Path("financial_terms_embeddings.pkl")

        # Initialize models
        self._init_models()
        
        # Load dictionary and embeddings
        self.financial_terms = self.load_financial_dictionary()
        self.financial_terms_embeddings = self.load_or_create_embeddings()

    def _init_models(self) -> None:
        """Initialize all required models"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained("upskyy/kf-deberta-multitask")
            self.model = AutoModel.from_pretrained("upskyy/kf-deberta-multitask").to(self.device)
            self.kiwi = Kiwi()
            self.keybert_model = KeyBERT("multi-qa-mpnet-base-cos-v1")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def load_financial_dictionary(self) -> List[str]:
        """Load and clean financial terms dictionary"""
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Clean and deduplicate terms
            terms = data.get('terms', [])
            cleaned_terms = {re.sub(r'\(.*?\)', '', term).strip() for term in terms}
            cleaned_terms = {term for term in cleaned_terms if len(term) >= self.config.min_keyword_length}
            
            logger.info(f"Loaded {len(cleaned_terms)} financial terms")
            return list(cleaned_terms)
        except Exception as e:
            logger.error(f"Error loading dictionary: {e}")
            return []

    def load_or_create_embeddings(self) -> torch.Tensor:
        """Load cached embeddings or create new ones"""
        if self.config.cache_embeddings and self.embedding_cache_path.exists():
            try:
                with open(self.embedding_cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if set(cached_data['terms']) == set(self.financial_terms):
                        logger.info("Using cached embeddings")
                        return cached_data['embeddings']
            except Exception as e:
                logger.warning(f"Error loading cached embeddings: {e}")

        logger.info("Creating new embeddings...")
        embeddings = self._create_embeddings(self.financial_terms)
        
        if self.config.cache_embeddings:
            with open(self.embedding_cache_path, 'wb') as f:
                pickle.dump({'terms': self.financial_terms, 'embeddings': embeddings}, f)
        
        return embeddings

    def _create_embeddings(self, terms: List[str]) -> torch.Tensor:
        """Create embeddings for given terms in batches"""
        if not terms:
            return torch.empty(0, self.model.config.hidden_size)

        embeddings = []
        for i in range(0, len(terms), self.config.batch_size):
            batch_terms = terms[i:i + self.config.batch_size]
            inputs = self.tokenizer(batch_terms, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu())
            
        torch.cuda.empty_cache()
        return torch.cat(embeddings)

    def extract_article_text(self) -> Tuple[str, str]:
        """Extract title and content from article HTML"""
        try:
            with open(self.article_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
            
            title = soup.title.string if soup.title else ""
            content = soup.find(id="dic_area")
            content_text = content.get_text(strip=True) if content else ""
            
            return title.strip(), content_text.strip()
        except Exception as e:
            logger.error(f"Error extracting article text: {e}")
            return "", ""

    def extract_keywords(self, methods: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Extract keywords using specified methods

        Args:
            methods: List of methods to use ['kf-deberta', 'keybert', 'tfidf']

        Returns:
            Dictionary containing extracted keywords by method
        """
        methods = methods or ['kf-deberta', 'keybert', 'tfidf']
        title, content = self.extract_article_text()
        article_text = f"{title} {content}"
        
        # Extract nouns once for reuse
        nouns = self._extract_nouns(article_text)
        noun_embeddings = self._create_embeddings(nouns)

        results = {}
        
        # Extract keywords using each method
        for method in methods:
            try:
                if method == 'kf-deberta':
                    results['kf-deberta'] = self._extract_similarity_keywords(
                        noun_embeddings, nouns
                    )
                elif method == 'keybert':
                    results['keybert'] = self._extract_keybert_keywords(article_text)
                elif method == 'tfidf':
                    results['tfidf'] = self._extract_tfidf_keywords(article_text)
            except Exception as e:
                logger.error(f"Error extracting keywords with {method}: {e}")
                results[method] = []

        # Combine all unique keywords
        all_keywords = set()
        for keywords in results.values():
            all_keywords.update(keywords)
        
        results['all_keywords'] = list(all_keywords)
        return results

    def _extract_nouns(self, text: str) -> List[str]:
        """Extract nouns using Kiwi morphological analyzer"""
        try:
            result = self.kiwi.analyze(text)
            nouns = [token[0] for token in result[0][0] if token[1].startswith('N')]
            return [noun for noun in nouns if len(noun) >= self.config.min_keyword_length]
        except Exception as e:
            logger.error(f"Error extracting nouns: {e}")
            return []

    def _extract_similarity_keywords(
        self,
        noun_embeddings: torch.Tensor,
        nouns: List[str]
    ) -> List[str]:
        """Extract keywords based on similarity to financial terms"""
        if noun_embeddings.dim() == 1:
            noun_embeddings = noun_embeddings.unsqueeze(0)
        if self.financial_terms_embeddings.dim() == 1:
            self.financial_terms_embeddings = self.financial_terms_embeddings.unsqueeze(0)

        similar_keywords = []
        for i, noun_embedding in enumerate(noun_embeddings):
            scores = util.pytorch_cos_sim(
                noun_embedding.unsqueeze(0),
                self.financial_terms_embeddings
            )[0]
            if max(scores).item() > self.config.similarity_threshold:
                similar_keywords.append(nouns[i])

        return sorted(
            similar_keywords,
            key=lambda x: len(x),
            reverse=True
        )[:self.config.top_n]

    def _extract_keybert_keywords(self, text: str) -> List[str]:
        """Extract keywords using KeyBERT"""
        keybert_keywords = [
            kw[0] for kw in self.keybert_model.extract_keywords(
                text,
                top_n=self.config.top_n,
                stop_words=None
            )
        ]
        return [
            kw for kw in self._extract_nouns(' '.join(keybert_keywords))
            if len(kw) >= self.config.min_keyword_length
        ]

    def _extract_tfidf_keywords(self, text: str) -> List[str]:
        """Extract keywords using TF-IDF"""
        nouns = self._extract_nouns(text)
        if not nouns:
            return []

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([' '.join(nouns)])
        terms = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        indices = np.argsort(scores)[::-1][:self.config.top_n]
        
        return [terms[idx] for idx in indices]

def format_keywords_output(keywords_dict: Dict[str, List[str]]) -> str:
    """Format extracted keywords for display"""
    output = ["\n추출된 키워드:"]
    for method, keywords in keywords_dict.items():
        method_name = "전체 키워드" if method == "all_keywords" else f"{method} 키워드"
        unique_keywords = sorted(set(keywords))
        output.extend([
            f"\n- {method_name}:",
            ", ".join(unique_keywords) if unique_keywords else "없음"
        ])
    return "\n".join(output)

def main():
    """Main function for keyword extraction"""
    config = KeywordExtractionConfig(
        top_n=10,
        similarity_threshold=0.6,
        min_keyword_length=2,
        batch_size=128,
        cache_embeddings=True
    )

    article_path = "sample1.html"
    dictionary_path = "econ_dictionary.json"

    try:
        extractor = KeywordExtractor(article_path, dictionary_path, config)
        keywords = extractor.extract_keywords()
        print(format_keywords_output(keywords))
    except Exception as e:
        logger.error(f"Error in keyword extraction: {e}")

if __name__ == '__main__':
    main()