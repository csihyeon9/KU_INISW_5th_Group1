## Hypergraph Neural Networks for News Recommendation

Created by S. H. Kwan, K. H. Kim, S. H. Park, J. H. Lee, C. H. Cho, S. H. Cha from Korea University INISW Academy, 5th  

### **Usage**
This project uses Hypergraph Neural Networks (HGNN) to recommend news articles based on their relationships modeled using hypergraphs.

To get started:
1. Create a Python virtual environment.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### **Training and Evaluating HGNN**
Train the HGNN model for news recommendation using the following command:
```
python train.py
```
- **Input**: `train.py` uses a pre-defined dataset (e.g., `./home/kuinisw5/data/EconFinNews_Raw.json`).
- **Output**: A trained HGNN model is saved as `hgnn_recommender_model.pth` for later use in recommendations.

### **Personalized Recommendation**
After training the model, perform personalized recommendations by processing new articles stored in the `personalization_data` folder. For example:
```
python recommendation.py
```
- **Input**: `./personalization_data/New.json`, containing details of new articles to be recommended.
- **Output**: Top recommended articles and related keywords printed to the console.

---

### **Code Description**

#### **train.py**
`train.py` is responsible for:
1. **Loading and Preprocessing the Data**:
   - The dataset is loaded from a JSON file (`./home/kuinisw5/data/EconFinNews_Raw.json`).
   - Features, labels, and train-test splits are created for hypergraph modeling.

2. **Constructing the Hypergraph**:
   - A hypergraph is created based on shared keywords between articles.
   - The hypergraph incidence matrix \( H \) and normalized graph \( G \) are generated.

3. **Training HGNN**:
   - **Model**: `HGNNRecommender` is used to train on the hypergraph \( G \) and input features.
   - **Optimization**:
     - Loss: Cross-entropy for classification tasks.
     - Optimizer: AdamW.
     - Learning Rate Scheduler: CosineAnnealingLR for learning rate decay.
   - **Output**:
     - The model is trained for the configured number of epochs.
     - The trained model is saved as `hgnn_recommender_model.pth`.

4. **Key Training Steps**:
   - The hypergraph captures relationships between articles based on shared keywords.
   - The HGNN learns to represent articles in a feature space where similar articles are closer.

#### **recommendation.py**
`recommendation.py` performs personalized recommendations using the trained HGNN model.

1. **Model Initialization**:
   - The HGNN model is loaded from the saved checkpoint (`hgnn_recommender_model.pth`).
   - Precomputed document embeddings are generated for all articles in the dataset.

2. **New Article Processing**:
   - New articles are loaded from `./personalization_data/New.json`.
   - For each new article:
     - It is added to the hypergraph, and the hypergraph is updated.
     - The new article’s embedding is computed using the HGNN model.

3. **Recommendation**:
   - **Cosine Similarity**:
     - The new article's embedding is compared to the precomputed embeddings of existing articles.
     - The top `k` most similar articles are selected for recommendation.
   - **Keyword Extraction**:
     - Keywords from the recommended articles are aggregated to provide related keyword suggestions.

4. **Output**:
   - The recommended articles are displayed with their URLs, similarity scores, and associated keywords.
   - A list of related keywords is provided for further insight.

---

### **Example Input and Output**

#### **Input**:
- **Dataset**: `EconFinNews_Raw.json` containing news articles and their keywords.
- **New Articles**: `./personalization_data/New.json` containing:
  ```json
  [
      {
          "url": "https://n.news.naver.com/mnews/article/003/0012920303",
          "relation_type": "생활경제",
          "keywords": [
              "비트코인",
              "가상자산",
              "알트코인"
          ]
      }
  ]
  ```

#### **Output**:
- **Console Output**:
  ```
  Processing new article: https://n.news.naver.com/mnews/article/003/0012920303

  Top recommendations:
  1. URL: https://n.news.naver.com/mnews/article/015/0005048226, Similarity: 0.1994, Keywords: 스타벅스, 매출 감소, 주당 순이익
  2. URL: https://n.news.naver.com/mnews/article/088/0000911235, Similarity: 0.1811, Keywords: 삼성전자, 이재용, 조직 관료화

  Related Keywords:
  스타벅스, 매출 감소, 삼성전자, 주당 순이익
  ```

---

### **Citation**
This project is based on the following work:
```plaintext
@article{feng2018hypergraph,
  title={Hypergraph Neural Networks},
  author={Feng, Yifan and You, Haoxuan and Zhang, Zizhao and Ji, Rongrong and Gao, Yue},
  journal={AAAI 2019},
  year={2018}
}
```

### **License**
Our code is released under MIT License (see LICENSE file for details).
