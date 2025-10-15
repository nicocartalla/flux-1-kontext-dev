"""
Ray Serve entrypoint que usa la estructura modular existente.
Mantiene toda la arquitectura de src/* pero agrega decoradores de Ray Serve.
"""
import os

# Silenciar el warning de Ray sobre accelerator env vars
# El APIGateway no necesita GPU, solo el modelo
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys

# Ray Serve imports
from ray import serve
from ray.serve.handle import DeploymentHandle

# Imports de nuestra estructura existente (absolutos para Ray Serve)
from src.config.settings import get_settings, get_model_capabilities, get_ray_serve_settings
from src.model.qwen_image_edit import QwenImageEditModel as BaseQwenModel
from src.service.image_service import ImageService
from src.api import images
from src.api.dependencies import set_image_service

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
ray_settings = get_ray_serve_settings()


# ============================================================================
# Model Deployment (con Ray Serve)
# ============================================================================

@serve.deployment(**ray_settings.get_model_deployment_config())
class QwenImageEditModelRay(BaseQwenModel):
    """
    Wrapper de Ray Serve sobre nuestra implementación existente.
    Hereda toda la funcionalidad de src/model/qwen_image_edit.py
    """
    
    def __init__(self):
        logger.info("🚀 Inicializando QwenImageEditModelRay con Ray Serve")
        # Llamar al __init__ de la clase base
        super().__init__()
        # La clase base ya tiene load_model, warmup, etc.
        self.load_model()
        
        if settings.enable_warmup:
            self.warmup()
        
        logger.info("✅ Modelo listo para servir con Ray")


# ============================================================================
# Crear FastAPI app ANTES del deployment
# ============================================================================

app = FastAPI(
    title=settings.app_name + " - Ray Serve",
    version=settings.app_version,
    description="""
    API compatible con OpenAI para servir modelos diffusers.
    Powered by Ray Serve para auto-scaling y gestión eficiente de GPUs.
    
    ## Endpoints Disponibles
    
    - **POST /v1/images/edits**: Edita imágenes usando prompts (Image Editing) ✅ 
    
    ## Framework
    
    - **Ray Serve**: Auto-scaling, GPU pooling, batching dinámico
    """,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Imports necesarios para los endpoints
# ============================================================================

from fastapi import UploadFile, File, Form, HTTPException, Request
from typing import Optional
import time
import base64
from io import BytesIO
from src.domain.openai_models import ImageResponse, ImageObject, ResponseFormat, ImageEditRequest


# ============================================================================
# Variable global para compartir ImageService
# ============================================================================

# Usamos un dict con lock para thread-safety
import threading
_service_lock = threading.Lock()
_service_registry = {}


def _register_service(replica_id: str, service):
    """Registra un ImageService para una réplica específica."""
    with _service_lock:
        _service_registry[replica_id] = service
        logger.info(f"✅ Servicio registrado para réplica {replica_id}")


def _get_service():
    """Obtiene el ImageService de la réplica actual."""
    try:
        # En Ray Serve, cada réplica tiene su propio proceso
        # Usamos el PID del proceso como identificador
        import os
        replica_id = str(os.getpid())
        
        with _service_lock:
            service = _service_registry.get(replica_id)
            if service:
                logger.debug(f"Servicio encontrado para replica {replica_id}")
                return service
            else:
                logger.warning(f"No hay servicio para replica {replica_id}. Registry: {list(_service_registry.keys())}")
                return None
    except Exception as e:
        logger.error(f"Error obteniendo servicio: {e}", exc_info=True)
        return None


# ============================================================================
# Endpoints a nivel de módulo (ANTES del deployment)
# ============================================================================

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
        "framework": "Ray Serve",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "framework": "ray-serve"}


@app.post(
    f"{settings.api_prefix}/images/edits",
    response_model=ImageResponse,
    summary="Edita una imagen usando un prompt (OpenAI compatible)"
)
async def create_image_edit(
    prompt: str = Form(..., description="Instrucciones de edición"),
    image: UploadFile = File(..., description="Imagen a editar"),
    mask: Optional[UploadFile] = File(None, description="Máscara de edición (opcional)"),
    model: Optional[str] = Form(None),
    n: Optional[int] = Form(1),
    size: Optional[str] = Form(None),
    response_format: Optional[str] = Form("b64_json"),
    num_inference_steps: Optional[int] = Form(50),
    guidance_scale: Optional[float] = Form(7.5),
    true_cfg_scale: Optional[float] = Form(4.0),
    negative_prompt: Optional[str] = Form(" "),
    seed: Optional[int] = Form(None),
    user: Optional[str] = Form(None),
):
    """
    Edita una imagen según las instrucciones del prompt.
    Compatible con OpenAI API: POST /v1/images/edits
    """
    # Obtener el servicio
    image_service = _get_service()
    logger.info(f"🔍 image_service: {image_service}")
    
    if image_service is None:
        logger.error("❌ image_service not initialized")
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if not capabilities.supports_image_edit:
        raise HTTPException(
            status_code=501,
            detail={
                "error": {
                    "message": "El modelo actual no soporta edición de imágenes",
                    "type": "not_supported_error",
                    "code": "model_not_supported"
                }
            }
        )
    
    try:
        logger.info(f"Editando imagen con prompt: {prompt[:50]}...")
        
        # Leer imagen
        image_bytes = await image.read()
        
        # Crear request object
        edit_request = ImageEditRequest(
            prompt=prompt,
            image="",
            mask="" if mask is None else "",
            model=model,
            n=n,
            size=size,
            response_format=ResponseFormat(response_format) if response_format else ResponseFormat.B64_JSON,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            seed=seed,
            user=user,
        )
        
        # Usar el ImageService (ahora es async)
        images_result = await image_service.edit_image(edit_request, image_file=image_bytes)
        
        # Construir respuesta
        image_objects = []
        for img in images_result:
            b64_img = image_service.encode_image(img, ResponseFormat.B64_JSON)
            image_objects.append(ImageObject(b64_json=b64_img))
        
        return ImageResponse(
            created=int(time.time()),
            data=image_objects
        )
        
    except Exception as e:
        logger.error(f"Error editando imagen: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "internal_error"
                }
            }
        )


# ============================================================================
# API Gateway Deployment (con Ray Serve) - DESPUÉS de definir endpoints
# ============================================================================

@serve.deployment(**ray_settings.get_api_deployment_config())
@serve.ingress(app)
class APIGateway:
    """
    Gateway que maneja requests HTTP usando nuestra estructura existente.
    Utiliza el ImageService para toda la lógica de negocio.
    """
    
    def __init__(self, model_handle: DeploymentHandle):
        logger.info("🌐 Inicializando APIGateway con Ray Serve")
        try:
            import os
            self.replica_id = str(os.getpid())
            logger.info(f"📍 Replica ID (PID): {self.replica_id}")
            
            self.model_handle = model_handle
            
            # Crear un wrapper del modelo para que funcione con nuestro ImageService
            self.model_wrapper = ModelHandleWrapper(model_handle)
            logger.info("✅ ModelHandleWrapper creado")
            
            # Usar nuestro ImageService existente
            self.image_service = ImageService(self.model_wrapper)
            logger.info("✅ ImageService creado")
            
            # Registrar el servicio globalmente usando el PID
            _register_service(self.replica_id, self.image_service)
            
            logger.info("✅ APIGateway completamente listo")
        except Exception as e:
            logger.error(f"❌ Error inicializando APIGateway: {e}", exc_info=True)
            raise


# ============================================================================
# Model Handle Wrapper
# ============================================================================

class ModelHandleWrapper:
    """
    Wrapper que adapta el DeploymentHandle de Ray para funcionar
    con nuestra interfaz BaseDiffusionModel existente.
    """
    
    def __init__(self, model_handle: DeploymentHandle):
        self.model_handle = model_handle
    
    def supports_image_edit(self) -> bool:
        return True
    
    def supports_text_to_image(self) -> bool:
        return False
    
    def supports_image_variation(self) -> bool:
        return False
    
    async def edit_image(self, image, prompt, mask=None, **kwargs):
        """
        Llama al modelo deployment de Ray de forma asíncrona.
        Convierte entre formato PIL y bytes según necesario.
        """
        from PIL import Image
        from io import BytesIO
        
        # Convertir PIL Image a bytes si es necesario
        if isinstance(image, Image.Image):
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
        else:
            image_bytes = image
        
        # Llamar al modelo remoto (esto es asíncrono con Ray)
        # El modelo ya devuelve una lista de imágenes
        result_images = await self.model_handle.edit_image.remote(
            image=image,
            prompt=prompt,
            mask=mask,
            **kwargs
        )
        
        return result_images


# ============================================================================
# Entrypoint para Ray Serve
# ============================================================================

# Crear el deployment del modelo
model_deployment = QwenImageEditModelRay.bind()

# Crear el deployment del gateway pasándole el handle del modelo
entrypoint = APIGateway.bind(model_deployment)

