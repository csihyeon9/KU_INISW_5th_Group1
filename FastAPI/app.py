# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import datetime
import pytz

app = FastAPI()

class InputText(BaseModel):
    text: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/service")
async def main(input: InputText):
    local_tz = pytz.timezone("Asia/Seoul")

    print("request_time:", datetime.datetime.now(local_tz))
    print("received_text:\n", input.text)

    return {
        "text": input.text,
        "response_time": datetime.datetime.now(local_tz).isoformat()
    }