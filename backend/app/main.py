
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.papers import router as papers_router
from backend.app.services.search_manager import search_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    search_manager.load_indexes()
    yield


app = FastAPI(
    title="PaperMind",
    description="Personal Research Paper Intelligence Engine",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(papers_router)


@app.get("/")
def root():
    return {
        "message": "Welcome to PaperMind",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "search_indexes_loaded": len(search_manager.stores),
    }
