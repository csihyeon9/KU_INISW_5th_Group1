# database.py
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from typing import Optional, List
import datetime
import os


class Database:
    client: Optional[AsyncIOMotorClient] = None
    article_collection = None
    connected: bool = False

    @classmethod
    async def connect_db(cls):
        try:
            # docker-compose.yml
            # mongo_uri = os.getenv('MONGO_URI', 'mongodb://mongodb:27017/finwise')
            # cls.client = AsyncIOMotorClient(mongo_uri)
            # cls.article_collection = cls.client.finwise.articles

            # server standalone
            cls.client = AsyncIOMotorClient("mongodb://finwise.p-e.kr:27017")
            cls.article_collection = cls.client.articles_db.articles

            # local
            # cls.client = AsyncIOMotorClient("mongodb://localhost:27017")
            # cls.article_collection = cls.client.articles_db.articles

            # 연결 테스트
            await cls.client.admin.command('ping')
            cls.connected = True
            print("MongoDB에 성공적으로 연결되었습니다!")
        except Exception as e:
            print(f"MongoDB 연결 중 오류 발생: {e}")
            cls.connected = False
            raise

    @classmethod
    async def close_db(cls):
        if cls.client and cls.connected:
            try:
                await cls.client.close()
                cls.connected = False
                print("MongoDB 연결이 안전하게 종료되었습니다.")
            except Exception as e:
                print(f"MongoDB 연결 종료 중 오류 발생: {e}")
        else:
            print("MongoDB 클라이언트가 없거나 이미 연결이 종료되었습니다.")

    @classmethod
    async def save_article(cls, article_data: dict) -> str:
        if not cls.connected:
            await cls.connect_db()

        article = {
            "title": article_data["title"],
            "date": article_data["date"],
            "url": article_data["url"],
            "content": article_data["content"],
            "hypergraph_data": article_data["hypergraph_data"],
            "recommendations": article_data["recommendations"],
            "created_at": datetime.datetime.utcnow()
        }
        result = await cls.article_collection.insert_one(article)
        return str(result.inserted_id)

    @classmethod
    async def get_all_articles(cls) -> List[dict]:
        if not cls.connected:
            await cls.connect_db()

        cursor = cls.article_collection.find({})
        articles = await cursor.to_list(length=None)
        for article in articles:
            article["_id"] = str(article["_id"])
        return articles

    @classmethod
    async def delete_article(cls, article_id: str) -> bool:
        if not cls.connected:
            await cls.connect_db()

        try:
            object_id = ObjectId(article_id)
            result = await cls.article_collection.delete_one({"_id": object_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting article: {str(e)}")
            return False