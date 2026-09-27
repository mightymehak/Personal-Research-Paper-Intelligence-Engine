from fastapi import FastAPI

from backend.app.api.papers import router as papers_router


app = FastAPI(
    title="PaperMind",
    description="Personal Research Paper Intelligence Engine",
    version="0.1.0"
)


app.include_router(papers_router)


@app.get("/")
def root():
    return {
        "message": "Welcome to PaperMind",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }