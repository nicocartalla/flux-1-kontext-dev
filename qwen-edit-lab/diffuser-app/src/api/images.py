"""
Router de FastAPI para endpoints de imágenes (OpenAI compatible).
Define TODOS los endpoints posibles, pero solo habilita los que el modelo soporta.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import Optional
import time
import logging

from ..domain.openai_models import (
    ImageResponse,
    ImageObject,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageVariationRequest,
    ResponseFormat,
    ErrorResponse,
    ErrorDetail,
)
from ..service.image_service import ImageService
from ..config.settings import get_model_capabilities

logger = logging.getLogger(__name__)

# Router
router = APIRouter(prefix="/images", tags=["images"])


def get_image_service() -> ImageService:
    """
    Dependency para obtener el servicio de imágenes.
    Debe ser configurado en el startup de la app.
    """
    from ..api.dependencies import image_service
    if image_service is None:
        raise HTTPException(
            status_code=503,
            detail="Servicio de imágenes no disponible"
        )
    return image_service


# ============================================================================
# POST /v1/images/generations - Text to Image
# ============================================================================

@router.post(
    "/generations",
    response_model=ImageResponse,
    responses={
        400: {"model": ErrorResponse},
        501: {"model": ErrorResponse},
    },
    summary="Genera imágenes desde texto (Text-to-Image)"
)
async def create_image_generation(
    request: ImageGenerationRequest,
    service: ImageService = Depends(get_image_service)
):
    """
    Genera una o más imágenes desde un prompt de texto.
    
    Compatible con OpenAI API: POST /v1/images/generations
    
    **Nota**: Este endpoint solo está habilitado si el modelo lo soporta.
    """
    capabilities = get_model_capabilities()
    
    if not capabilities.supports_text_to_image:
        raise HTTPException(
            status_code=501,
            detail={
                "error": {
                    "message": "El modelo actual no soporta generación de imágenes desde texto",
                    "type": "not_supported_error",
                    "code": "model_not_supported"
                }
            }
        )
    
    try:
        logger.info(f"Generando imagen con prompt: {request.prompt[:50]}...")
        
        # Generar imágenes
        images = service.generate_image(request)
        
        # Construir respuesta
        image_objects = []
        for img in images:
            if request.response_format == ResponseFormat.B64_JSON:
                b64_img = service.encode_image(img, ResponseFormat.B64_JSON)
                image_objects.append(ImageObject(b64_json=b64_img))
            else:
                # Por ahora solo soportamos b64_json
                # URL requeriría almacenamiento externo
                b64_img = service.encode_image(img, ResponseFormat.B64_JSON)
                image_objects.append(ImageObject(b64_json=b64_img))
        
        return ImageResponse(
            created=int(time.time()),
            data=image_objects
        )
        
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Error generando imagen: {e}", exc_info=True)
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
# POST /v1/images/edits - Image Editing
# ============================================================================

@router.post(
    "/edits",
    response_model=ImageResponse,
    responses={
        400: {"model": ErrorResponse},
        501: {"model": ErrorResponse},
    },
    summary="Edita una imagen usando un prompt (Image Editing)"
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
    service: ImageService = Depends(get_image_service)
):
    """
    Edita una imagen según las instrucciones del prompt.
    
    Compatible con OpenAI API: POST /v1/images/edits
    
    Este endpoint recibe la imagen como multipart/form-data.
    
    **Parámetros adicionales para diffusers:**
    - num_inference_steps: Pasos de inferencia (default: 50)
    - true_cfg_scale: CFG scale específico de Qwen (default: 4.0)
    - negative_prompt: Prompt negativo (default: " ")
    - seed: Seed para reproducibilidad
    """
    capabilities = get_model_capabilities()
    
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
        
        # Leer máscara si existe
        mask_bytes = None
        if mask:
            mask_bytes = await mask.read()
        
        # Crear request object
        request = ImageEditRequest(
            prompt=prompt,
            image="",  # Se pasa como bytes
            mask="" if mask_bytes is None else "",
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
        
        # Editar imagen
        images = service.edit_image(request, image_file=image_bytes)
        
        # Construir respuesta
        image_objects = []
        for img in images:
            if request.response_format == ResponseFormat.B64_JSON:
                b64_img = service.encode_image(img, ResponseFormat.B64_JSON)
                image_objects.append(ImageObject(b64_json=b64_img))
            else:
                b64_img = service.encode_image(img, ResponseFormat.B64_JSON)
                image_objects.append(ImageObject(b64_json=b64_img))
        
        return ImageResponse(
            created=int(time.time()),
            data=image_objects
        )
        
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": "invalid_image"
                }
            }
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
# POST /v1/images/variations - Image Variations
# ============================================================================

@router.post(
    "/variations",
    response_model=ImageResponse,
    responses={
        400: {"model": ErrorResponse},
        501: {"model": ErrorResponse},
    },
    summary="Crea variaciones de una imagen (Image Variation)"
)
async def create_image_variation(
    image: UploadFile = File(..., description="Imagen base"),
    model: Optional[str] = Form(None),
    n: Optional[int] = Form(1),
    response_format: Optional[str] = Form("b64_json"),
    size: Optional[str] = Form(None),
    num_inference_steps: Optional[int] = Form(50),
    strength: Optional[float] = Form(0.75),
    seed: Optional[int] = Form(None),
    user: Optional[str] = Form(None),
    service: ImageService = Depends(get_image_service)
):
    """
    Crea variaciones de una imagen proporcionada.
    
    Compatible con OpenAI API: POST /v1/images/variations
    
    **Nota**: Este endpoint solo está habilitado si el modelo lo soporta.
    """
    capabilities = get_model_capabilities()
    
    if not capabilities.supports_image_variation:
        raise HTTPException(
            status_code=501,
            detail={
                "error": {
                    "message": "El modelo actual no soporta variaciones de imagen",
                    "type": "not_supported_error",
                    "code": "model_not_supported"
                }
            }
        )
    
    try:
        logger.info("Creando variación de imagen...")
        
        # Leer imagen
        image_bytes = await image.read()
        
        # Crear request object
        request = ImageVariationRequest(
            image="",
            model=model,
            n=n,
            response_format=ResponseFormat(response_format) if response_format else ResponseFormat.B64_JSON,
            size=size,
            num_inference_steps=num_inference_steps,
            strength=strength,
            seed=seed,
            user=user,
        )
        
        # Crear variación
        images = service.create_variation(request, image_file=image_bytes)
        
        # Construir respuesta
        image_objects = []
        for img in images:
            if request.response_format == ResponseFormat.B64_JSON:
                b64_img = service.encode_image(img, ResponseFormat.B64_JSON)
                image_objects.append(ImageObject(b64_json=b64_img))
            else:
                b64_img = service.encode_image(img, ResponseFormat.B64_JSON)
                image_objects.append(ImageObject(b64_json=b64_img))
        
        return ImageResponse(
            created=int(time.time()),
            data=image_objects
        )
        
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": "invalid_image"
                }
            }
        )
    except Exception as e:
        logger.error(f"Error creando variación: {e}", exc_info=True)
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

