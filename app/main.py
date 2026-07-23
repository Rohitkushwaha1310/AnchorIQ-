from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import upload

app = FastAPI(title="InsightAI — Autonomous Data Analyst", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)


@app.get("/")
def root():
    return {"status": "ok", "service": "InsightAI"}
