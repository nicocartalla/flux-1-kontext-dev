"""
Configuración de la aplicación y modelos.
"""
from typing import Optional, Dict, Any
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


class RayServeSettings(BaseSettings):
    """
    Configuración específica para Ray Serve deployment.
    Todas estas settings se pueden sobrescribir con variables de entorno.
    """
    
    # ========================================================================
    # Model Deployment Settings
    # ========================================================================
    
    # GPU/CPU Resources
    model_num_gpus: float = 1.0  # Número de GPUs por replica del modelo
    model_num_cpus: float = 2.0  # CPUs por replica
    
    # Replicas
    model_num_replicas: int = 1  # Número inicial de replicas
    model_min_replicas: int = 1  # Mínimo de replicas (para autoscaling)
    model_max_replicas: int = 10  # Máximo de replicas (para autoscaling)
    
    # Autoscaling
    enable_autoscaling: bool = False  # Habilitar autoscaling
    target_num_ongoing_requests_per_replica: int = 2  # Requests objetivo por replica
    
    # Batching (opcional, para agrupar requests)
    enable_batching: bool = False  # Habilitar batching dinámico
    max_batch_size: int = 8  # Tamaño máximo del batch
    batch_wait_timeout_s: float = 0.1  # Tiempo de espera para formar batch
    
    # Concurrency
    max_ongoing_requests: int = 100  # Máximo de requests en curso por replica
    
    # ========================================================================
    # API Gateway Deployment Settings
    # ========================================================================
    
    # API Gateway Resources
    api_num_replicas: int = 1  # Replicas del API gateway
    api_num_cpus: float = 1.0  # CPUs para API gateway
    
    # ========================================================================
    # Health Check Settings
    # ========================================================================
    
    # Timeouts para health checks
    health_check_period_s: float = 10.0  # Cada cuánto verificar health
    health_check_timeout_s: float = 30.0  # Timeout para health check
    
    # ========================================================================
    # Deployment Strategy
    # ========================================================================
    
    # Graceful shutdown
    graceful_shutdown_timeout_s: float = 20.0  # Tiempo para shutdown graceful
    graceful_shutdown_wait_loop_s: float = 2.0  # Tiempo entre checks de shutdown
    
    class Config:
        env_prefix = "RAY_"
        case_sensitive = False
    
    def get_model_deployment_config(self) -> Dict[str, Any]:
        """
        Genera la configuración completa para el model deployment.
        Returns un dict que se puede usar directamente en @serve.deployment()
        """
        config = {
            "ray_actor_options": {
                "num_gpus": self.model_num_gpus,
                "num_cpus": self.model_num_cpus,
            },
            "max_ongoing_requests": self.max_ongoing_requests,
            "graceful_shutdown_timeout_s": self.graceful_shutdown_timeout_s,
            "graceful_shutdown_wait_loop_s": self.graceful_shutdown_wait_loop_s,
            "health_check_period_s": self.health_check_period_s,
            "health_check_timeout_s": self.health_check_timeout_s,
        }
        
        # Agregar autoscaling si está habilitado
        if self.enable_autoscaling:
            config["autoscaling_config"] = {
                "min_replicas": self.model_min_replicas,
                "max_replicas": self.model_max_replicas,
                "target_num_ongoing_requests_per_replica": self.target_num_ongoing_requests_per_replica,
            }
        else:
            config["num_replicas"] = self.model_num_replicas
        
        return config
    
    def get_api_deployment_config(self) -> Dict[str, Any]:
        """
        Genera la configuración para el API gateway deployment.
        """
        return {
            "num_replicas": self.api_num_replicas,
            "ray_actor_options": {
                "num_cpus": self.api_num_cpus,
            },
        }


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


@lru_cache()
def get_ray_serve_settings() -> RayServeSettings:
    """Obtiene la configuración de Ray Serve (singleton)."""
    return RayServeSettings()

