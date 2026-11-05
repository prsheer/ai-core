from fastapi import FastAPI

from app.api.embeddings import router as embeddings_router
from app.api.health import router as health_router


app = FastAPI(title="AI Core", version="0.1.0")
app.include_router(health_router)
app.include_router(embeddings_router)
