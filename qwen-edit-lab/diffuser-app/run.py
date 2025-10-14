#!/usr/bin/env python
"""
Script de entrada para ejecutar la aplicación.
"""
import sys
import os

# Agregar el directorio actual al path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

if __name__ == "__main__":
    # Importar settings después de ajustar el path
    from src.config.settings import get_settings
    
    settings = get_settings()
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level="info",
        reload=False,
    )

