"""
AI Travel Advisor - FastAPI Backend
Production-grade entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from routers import recommend, chat, destinations
from services.rag_service import rag_service

# Configure logging (real apps log everything)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan: runs on startup and shutdown.
    Real apps initialize DB connections, load models, etc. here.
    NOT inside route handlers (that would reload every request - rookie mistake).
    """
    logger.info("🚀 Starting AI Travel Advisor API...")
    await rag_service.initialize()
    logger.info("✅ RAG service initialized. Ready to serve!")
    yield
    # Cleanup on shutdown
    logger.info("🛑 Shutting down...")


app = FastAPI(
    title="AI Travel Advisor API",
    description="RAG-powered travel recommendation system for India",
    version="2.0.0",
    lifespan=lifespan
)

# CORS - allows frontend (React) to call this backend
# In production: replace "*" with your actual frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers (each router = one feature area)
app.include_router(recommend.router, prefix="/api/v1", tags=["Recommendations"])
app.include_router(chat.router,      prefix="/api/v1", tags=["Chat"])
app.include_router(destinations.router, prefix="/api/v1", tags=["Destinations"])


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Deployment platforms (Railway, Render) ping this to know if app is alive.
    """
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/")
async def root():
    return {"message": "AI Travel Advisor API v2.0", "docs": "/docs"}