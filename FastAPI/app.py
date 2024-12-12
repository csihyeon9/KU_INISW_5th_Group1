# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import datetime
import pytz
from database import Database
from graph_connector import GraphConnector

from nlp_processor import NLPProcessor
from relation_processor import RelationProcessor
from src.give import get_word_definition
# from src.relation_extractor import cleanup
# from recommend import ArticleRecommender
from src.fetch_content import fetch_content
from src.pkm_processor import PKMProcessor

app = FastAPI()

class InputText(BaseModel):
    # text: str = None
    url: str = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp_processor = NLPProcessor()
relation_processor = RelationProcessor(
    model_path='results/models/hgnn_model.pth',
    pmi_path='data/pairwise_pmi_values3.json'
)
# recommender = ArticleRecommender()

@app.post("/service")
async def main(input: InputText):
    local_tz = pytz.timezone("Asia/Seoul")
    title = ""
    date = ""
    url = ""
    content = ""

    print("request_time:", datetime.datetime.now(local_tz))

    if input.url:
        print("\nurl입력: ", input.url)
        article_data = fetch_content(input.url)
        print("\n크롤링 완료\n")

        if isinstance(article_data, dict):
            content = article_data.get("content", "No content found.")
            title = article_data.get("title", "No title found.")
            date = article_data.get("date", "No date found.")
            url = article_data.get("url", input.url)

    print("received_title:\n", title)
    print("received_date:\n", date)
    print("received_url:\n", url)
    print("received_text:\n", content)

    try:
        # 1. NLP 처리 및 그래프 생성
        graph_data = nlp_processor.process_text(content)
        print("\nresult:", graph_data)

        # 2. 관계 분류
        enhanced_graph = relation_processor.classify_relations(graph_data)
        print("\nenhanced result:", enhanced_graph)

        # 3. 고립 그래프 연결
        connector = GraphConnector(enhanced_graph, content)  # content 전달
        final_graph = connector.connect_isolated_graphs(relation_processor)
        print("\nfinal result:", final_graph)

        # 키워드 정의 처리
        cur_unique_keywords = [node["id"] for node in enhanced_graph["nodes"]]
        cur_unique_keywords = list(set(cur_unique_keywords))

        # cur_unique_keywords의 정의를 제공해야 함.
        try:
            definitions = get_word_definition(cur_unique_keywords, "./data/json/dictionary.json")
            if definitions is None:
                definitions = {}  # 정의를 찾지 못한 경우 빈 딕셔너리 반환
            print(f"정의 결과: {definitions}")
        except Exception as e:
            print(f"정의 검색 중 오류 발생: {str(e)}")
            definitions = {}  # 오류 발생 시 빈 딕셔너리로 처리

        # recommendations = recommender.recommend(cur_unique_keywords)
        # print("\nrecommendations:", recommendations)

        return JSONResponse({
            "hypergraph_data": final_graph,
            # "hypergraph_data": enhanced_graph,
            # "recommendations": recommendations,
            "response_time": datetime.datetime.now(local_tz).isoformat(),
            "title": title,
            "date": date,
            "url": url,
            "content": content,
            "definitions": definitions
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_db_client():
    await Database.connect_db()

@app.on_event("shutdown")
async def shutdown_event():
    await Database.close_db()
    nlp_processor.cleanup()

@app.post("/save_article")
async def save_article(article_data: dict):
    article_id = await Database.save_article(article_data)
    print("\n저장 성공")
    return {"message": "Article saved successfully", "article_id": article_id}

@app.get("/articles")
async def get_articles():
    try:
        # 1. 기존 문서들 가져오기
        articles = await Database.get_all_articles()
        if not articles:
            return []

        # return articles

        # 2. PKM 프로세서로 문서 간 연결 처리
        try:
            pkm_processor = PKMProcessor(relation_processor)
            integrated_articles = pkm_processor.process_articles(articles)
            print("\nintegrated_articles:", integrated_articles)
            return integrated_articles
        except Exception as e:
            print(f"Error processing articles: {str(e)}")
            # 오류 발생시 원본 articles 반환
            return articles

    except Exception as e:
        print(f"Error in get_articles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/articles/{article_id}")
async def delete_article(article_id: str):
    try:
        success = await Database.delete_article(article_id)
        if success:
            print("\n삭제 성공")
            return {"message": f"Article {article_id} successfully deleted"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Article {article_id} not found"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting article: {str(e)}"
        )