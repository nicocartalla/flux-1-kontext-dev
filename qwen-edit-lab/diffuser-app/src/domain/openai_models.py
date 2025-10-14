"""
OpenAI API compatible models for image generation endpoints.
Define todos los modelos según el estándar de OpenAI para permitir 
control total de los parámetros, pero cada modelo implementará solo los que necesite.
"""
from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from enum import Enum


class ImageSize(str, Enum):
    """Tamaños estándar de imagen según OpenAI."""
    SIZE_256 = "256x256"
    SIZE_512 = "512x512"
    SIZE_1024 = "1024x1024"
    SIZE_1792_1024 = "1792x1024"
    SIZE_1024_1792 = "1024x1792"


class ResponseFormat(str, Enum):
    """Formato de respuesta para las imágenes."""
    URL = "url"
    B64_JSON = "b64_json"


# ============================================================================
# Image Generation Endpoints (POST /v1/images/generations)
# ============================================================================

class ImageGenerationRequest(BaseModel):
    """Request para generar imágenes desde texto (text-to-image)."""
    prompt: str = Field(..., description="Descripción de la imagen a generar")
    model: Optional[str] = Field(None, description="Modelo a usar")
    n: Optional[int] = Field(1, ge=1, le=10, description="Número de imágenes a generar")
    quality: Optional[Literal["standard", "hd"]] = Field("standard", description="Calidad de la imagen")
    response_format: Optional[ResponseFormat] = Field(ResponseFormat.URL, description="Formato de respuesta")
    size: Optional[ImageSize] = Field(ImageSize.SIZE_1024, description="Tamaño de la imagen")
    style: Optional[Literal["vivid", "natural"]] = Field("vivid", description="Estilo de la imagen")
    user: Optional[str] = Field(None, description="Usuario único para tracking")
    
    # Parámetros adicionales para diffusers
    num_inference_steps: Optional[int] = Field(50, ge=1, le=150, description="Pasos de inferencia")
    guidance_scale: Optional[float] = Field(7.5, ge=0.0, le=20.0, description="Escala de guidance")
    negative_prompt: Optional[str] = Field(None, description="Prompt negativo")
    seed: Optional[int] = Field(None, description="Seed para reproducibilidad")


# ============================================================================
# Image Edit Endpoints (POST /v1/images/edits)
# ============================================================================

class ImageEditRequest(BaseModel):
    """Request para editar imágenes con texto (image-to-image editing)."""
    # Nota: La imagen se envía como multipart/form-data, no en JSON
    # Estos campos serán parseados desde el form
    prompt: str = Field(..., description="Instrucciones de edición")
    image: str = Field(..., description="Imagen a editar (base64 o file)")
    mask: Optional[str] = Field(None, description="Máscara de edición (base64 o file)")
    model: Optional[str] = Field(None, description="Modelo a usar")
    n: Optional[int] = Field(1, ge=1, le=10, description="Número de imágenes a generar")
    size: Optional[ImageSize] = Field(None, description="Tamaño de salida")
    response_format: Optional[ResponseFormat] = Field(ResponseFormat.URL, description="Formato de respuesta")
    user: Optional[str] = Field(None, description="Usuario único para tracking")
    
    # Parámetros adicionales para diffusers
    num_inference_steps: Optional[int] = Field(50, ge=1, le=150, description="Pasos de inferencia")
    guidance_scale: Optional[float] = Field(7.5, ge=0.0, le=20.0, description="Escala de guidance")
    true_cfg_scale: Optional[float] = Field(4.0, ge=0.0, le=20.0, description="True CFG scale (Qwen specific)")
    negative_prompt: Optional[str] = Field(" ", description="Prompt negativo")
    seed: Optional[int] = Field(None, description="Seed para reproducibilidad")


# ============================================================================
# Image Variation Endpoints (POST /v1/images/variations)
# ============================================================================

class ImageVariationRequest(BaseModel):
    """Request para crear variaciones de una imagen."""
    # La imagen se envía como multipart/form-data
    image: str = Field(..., description="Imagen base (base64 o file)")
    model: Optional[str] = Field(None, description="Modelo a usar")
    n: Optional[int] = Field(1, ge=1, le=10, description="Número de variaciones")
    response_format: Optional[ResponseFormat] = Field(ResponseFormat.URL, description="Formato de respuesta")
    size: Optional[ImageSize] = Field(None, description="Tamaño de salida")
    user: Optional[str] = Field(None, description="Usuario único para tracking")
    
    # Parámetros adicionales para diffusers
    num_inference_steps: Optional[int] = Field(50, ge=1, le=150, description="Pasos de inferencia")
    strength: Optional[float] = Field(0.75, ge=0.0, le=1.0, description="Fuerza de la variación")
    seed: Optional[int] = Field(None, description="Seed para reproducibilidad")


# ============================================================================
# Response Models
# ============================================================================

class ImageObject(BaseModel):
    """Objeto individual de imagen en la respuesta."""
    url: Optional[str] = Field(None, description="URL de la imagen generada")
    b64_json: Optional[str] = Field(None, description="Imagen en base64")
    revised_prompt: Optional[str] = Field(None, description="Prompt revisado/procesado")


class ImageResponse(BaseModel):
    """Respuesta estándar de OpenAI para endpoints de imágenes."""
    created: int = Field(..., description="Timestamp Unix de creación")
    data: List[ImageObject] = Field(..., description="Lista de imágenes generadas")


# ============================================================================
# Error Models
# ============================================================================

class ErrorDetail(BaseModel):
    """Detalle de error."""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Respuesta de error estándar de OpenAI."""
    error: ErrorDetail

