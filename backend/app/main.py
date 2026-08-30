from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .api.endpoints import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup routines
    settings.ensure_directories()
    print(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} initialized successfully.")
    yield
    # Shutdown routines
    print(f"👋 Shutting down {settings.APP_NAME}.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade, FAANG-caliber RAG engine for academic literature with hybrid search and page-level grounding.",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount endpoints under /api and also directly
app.include_router(api_router, prefix="/api", tags=["Research Mind API"])
app.include_router(api_router, tags=["Direct Routes"])


@app.get("/", tags=["System"])
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "docs_url": "/docs",
        "health_check": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=settings.DEBUG,
    )
