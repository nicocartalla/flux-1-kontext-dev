"""
Script de entrada para ejecutar la aplicación.
"""
import uvicorn
from src.config.settings import get_settings

if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level="info",
        reload=False,
    )

