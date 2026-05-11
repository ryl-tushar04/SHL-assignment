"""
SHL Assessment Recommender – FastAPI Service
Endpoints:
  GET  /health   → {"status": "ok"}
  POST /chat     → {reply, recommendations, end_of_conversation}
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ before anything else reads it

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from agent import run_agent
from scraper import load_or_scrape_catalog
from vector_store import CatalogVectorStore

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────
_vector_store: Optional[CatalogVectorStore] = None
_startup_time: float = 0.0


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vector_store, _startup_time
    t0 = time.time()
    logger.info("=== SHL Recommender starting up ===")

    # 1. Load or scrape catalog
    catalog = load_or_scrape_catalog()
    logger.info(f"Catalog loaded: {len(catalog)} assessments")

    # 2. Build or load vector store
    store = CatalogVectorStore(catalog)

    loaded = store.load()

    if not loaded:
        logger.error("Prebuilt vector store not found.")
        raise RuntimeError("Vector store missing. Build locally before deployment.")

    _vector_store = store

    _startup_time = time.time() - t0
    logger.info(f"Startup complete in {_startup_time:.1f}s")
    yield
    logger.info("=== SHL Recommender shutting down ===")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL assessments.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message content must not be empty")
        return v.strip()


class ChatRequest(BaseModel):
    messages: list[Message] = Field(
        ...,
        min_length=1,
        max_length=16,  # 8 turns (user + assistant each), matches server-side cap
        description="Full conversation history",
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, msgs: list[Message]) -> list[Message]:
        if not msgs:
            raise ValueError("messages must not be empty")
        # Last message must be from user
        if msgs[-1].role != "user":
            raise ValueError("Last message must be from the user")
        return msgs


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="Service not ready. Please retry.")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Hard turn cap: evaluator uses max 8 turns (user+assistant combined)
    if len(messages) > 16:
        return ChatResponse(
            reply="This conversation has reached its maximum length. Please start a new session.",
            recommendations=[],
            end_of_conversation=True,
        )

    try:
        result = run_agent(messages, _vector_store)
    except Exception as e:
        logger.exception(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail="Internal agent error")

    recs = [
        Recommendation(
            name=r["name"],
            url=r["url"],
            test_type=r.get("test_type", ""),
        )
        for r in result.get("recommendations", [])
    ]

    return ChatResponse(
        reply=result["reply"],
        recommendations=recs,
        end_of_conversation=result.get("end_of_conversation", False),
    )


# ── Error handlers ────────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )
