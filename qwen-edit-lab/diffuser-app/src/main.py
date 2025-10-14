"""
Aplicación principal FastAPI para servir modelos diffusers.
Compatible con OpenAI API.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys

from .config.settings import get_settings, get_model_capabilities
from .model.qwen_image_edit import QwenImageEditModel
from .service.image_service import ImageService
from .api import images
from .api.dependencies import set_image_service

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Obtener configuración
settings = get_settings()
capabilities = get_model_capabilities()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler para startup y shutdown."""
    # Startup
    logger.info("=" * 80)
    logger.info(f"Iniciando {settings.app_name} v{settings.app_version}")
    logger.info("=" * 80)
    
    # Cargar modelo
    logger.info(f"Modelo: {settings.model_id}")
    logger.info(f"Dispositivo: {settings.device}")
    logger.info(f"Dtype: {settings.torch_dtype}")
    
    # Instanciar modelo (por ahora solo Qwen Image Edit)
    # En el futuro, esto podría ser dinámico basado en configuración
    model = QwenImageEditModel()
    model.load_model()
    
    # Warmup si está habilitado
    if settings.enable_warmup:
        model.warmup()
    
    # Crear servicio
    service = ImageService(model)
    set_image_service(service)
    
    # Mostrar capacidades
    logger.info("=" * 80)
    logger.info("Capacidades del modelo:")
    logger.info(f"  - Text-to-Image: {capabilities.supports_text_to_image}")
    logger.info(f"  - Image Edit: {capabilities.supports_image_edit}")
    logger.info(f"  - Image Variation: {capabilities.supports_image_variation}")
    logger.info("=" * 80)
    logger.info(f"Servidor listo en http://{settings.host}:{settings.port}")
    logger.info(f"Documentación: http://{settings.host}:{settings.port}/docs")
    logger.info("=" * 80)
    
    yield
    
    # Shutdown (cleanup si es necesario)
    logger.info("Cerrando aplicación...")


# Crear aplicación FastAPI con lifespan
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    API compatible con OpenAI para servir modelos diffusers.
    
    ## Endpoints Disponibles
    
    Dependiendo de las capacidades del modelo cargado:
    
    - **POST /v1/images/generations**: Genera imágenes desde texto (Text-to-Image)
    - **POST /v1/images/edits**: Edita imágenes usando prompts (Image Editing) ✅ 
    - **POST /v1/images/variations**: Crea variaciones de imágenes (Image Variation)
    
    ✅ = Habilitado para el modelo actual (Qwen Image Edit)
    
    ## Parámetros Adicionales
    
    Además de los parámetros estándar de OpenAI, soportamos:
    - `num_inference_steps`: Número de pasos de inferencia
    - `true_cfg_scale`: Classifier-free guidance scale (Qwen specific)
    - `guidance_scale`: Guidance scale general
    - `negative_prompt`: Prompt negativo
    - `seed`: Seed para reproducibilidad
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Endpoint raíz con información del servicio."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "model": settings.model_id,
        "capabilities": {
            "text_to_image": capabilities.supports_text_to_image,
            "image_edit": capabilities.supports_image_edit,
            "image_variation": capabilities.supports_image_variation,
        },
        "endpoints": {
            "generations": f"{settings.api_prefix}/images/generations",
            "edits": f"{settings.api_prefix}/images/edits",
            "variations": f"{settings.api_prefix}/images/variations",
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# Incluir routers
app.include_router(images.router, prefix=settings.api_prefix)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level="info",
    )

