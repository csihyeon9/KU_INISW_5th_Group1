# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import datetime
import pytz
from database import Database

from nlp_processor import NLPProcessor
from src.give import get_word_definition
from src.relation_extractor import cleanup
from recommend import ArticleRecommender
from src.fetch_content import fetch_content

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
recommender = ArticleRecommender()

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
        result = nlp_processor.process_text(content)
        print("\nresult:", result)

        cur_unique_keywords = [node["id"] for node in result["nodes"]]
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

        recommendations = recommender.recommend(cur_unique_keywords)
        print("\nrecommendations:", recommendations)

        return JSONResponse({
            "hypergraph_data": result,
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
    cleanup()

@app.post("/save_article")
async def save_article(article_data: dict):
    article_id = await Database.save_article(article_data)
    print("\n저장 성공")
    return {"message": "Article saved successfully", "article_id": article_id}

@app.get("/articles")
async def get_articles():
    articles = await Database.get_all_articles()
    return articles

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