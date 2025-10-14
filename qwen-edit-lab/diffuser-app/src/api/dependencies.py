"""
Dependency injection para FastAPI.
Almacena las instancias globales de servicios.
"""
from typing import Optional
from ..service.image_service import ImageService

# Instancia global del servicio de imágenes
# Se inicializa en el startup de la aplicación
image_service: Optional[ImageService] = None


def set_image_service(service: ImageService):
    """Configura el servicio de imágenes global."""
    global image_service
    image_service = service


def get_image_service() -> ImageService:
    """Obtiene el servicio de imágenes."""
    if image_service is None:
        raise RuntimeError("Image service no inicializado")
    return image_service

