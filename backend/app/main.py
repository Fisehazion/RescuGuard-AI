"""
RescuGuard AI - FastAPI Application Factory & Server Entry Point
-----------------------------------------------------------------
Initializes the FastAPI application, registers middleware (CORS),
mounts API route controllers, and serves the static frontend command center.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.endpoints import router as api_router
from app.config import settings


def create_app() -> FastAPI:
    """
    Factory function to configure and instantiate the FastAPI application.
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Configure Cross-Origin Resource Sharing (CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API Version 1 endpoints
    app.include_router(api_router, prefix="/api/v1")

    # Mount static assets for the Emergency Command Center Dashboard
    if settings.FRONTEND_DIR.exists():
        app.mount(
            "/", 
            StaticFiles(directory=str(settings.FRONTEND_DIR), html=True), 
            name="frontend"
        )

    return app


app = create_app()