"""
FastAPI application entrypoint for Grammar Error Detection & Correction.
Configures CORS, lifespan model preloading, and route registration.
"""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.api.routes import router
from backend.app.models.grammar_model import GrammarModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for application startup and shutdown events."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}...")
    logger.info(f"Device: {settings.DEVICE} | Model: {settings.MODEL_NAME_OR_PATH}")
    
    # Preload model in background during startup for fast first request
    try:
        model = GrammarModel.get_instance()
        logger.info("Initializing model...")
        model.load()
    except Exception as e:
        logger.warning(f"Initial model preload deferred: {e}")

    yield

    logger.info("Shutting down application...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="End-to-end NLP system for Grammar Error Detection (GED) and Grammar Error Correction (GEC).",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)


@app.get("/")
def root():
    """Root info endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "online",
        "documentation": "/docs",
        "health": "/api/health",
        "endpoints": [
            "/api/correct",
            "/api/detect",
            "/api/candidates",
            "/api/evaluate",
            "/api/health",
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
