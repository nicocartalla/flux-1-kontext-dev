"""
Configuración de la aplicación y modelos.
"""
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Configuración general de la aplicación."""
    
    # API Settings
    app_name: str = "Diffuser API - OpenAI Compatible"
    app_version: str = "1.0.0"
    api_prefix: str = "/v1"
    
    # Model Settings
    model_id: str = "Qwen/Qwen-Image-Edit"
    model_name: str = "qwen-image-edit"
    torch_dtype: str = "bfloat16"  # bfloat16, float16, float32
    device: str = "cuda"  # cuda, cpu, mps
    
    # Default Generation Parameters
    default_num_inference_steps: int = 50
    default_guidance_scale: float = 7.5
    default_true_cfg_scale: float = 4.0
    default_negative_prompt: str = " "
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Warmup Settings
    enable_warmup: bool = False
    warmup_image_path: Optional[str] = None
    
    class Config:
        env_prefix = "DIFFUSER_"
        case_sensitive = False


class ModelCapabilities(BaseSettings):
    """
    Define qué capacidades tiene el modelo cargado.
    Esto permite que la API sepa qué endpoints habilitar.
    """
    supports_text_to_image: bool = False
    supports_image_edit: bool = True  # Qwen Image Edit solo soporta edición
    supports_image_variation: bool = False
    supports_inpainting: bool = False
    supports_mask: bool = False
    
    class Config:
        env_prefix = "MODEL_CAP_"


@lru_cache()
def get_settings() -> Settings:
    """Obtiene la configuración de la aplicación (singleton)."""
    return Settings()


@lru_cache()
def get_model_capabilities() -> ModelCapabilities:
    """Obtiene las capacidades del modelo (singleton)."""
    return ModelCapabilities()

