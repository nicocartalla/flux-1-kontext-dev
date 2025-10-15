"""
Clase base abstracta para todos los modelos de diffusion.
Define la interfaz que cada modelo debe implementar.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from PIL import Image


class BaseDiffusionModel(ABC):
    """
    Interfaz base para modelos de diffusion.
    Cada modelo implementa solo los métodos que soporta.
    """
    
    @abstractmethod
    def load_model(self) -> None:
        """Carga el modelo en memoria."""
        pass
    
    def warmup(self) -> None:
        """
        Realiza un warmup del modelo (opcional).
        Override en subclases si se necesita.
        """
        pass
    
    def supports_text_to_image(self) -> bool:
        """Indica si el modelo soporta text-to-image."""
        return False
    
    def supports_image_edit(self) -> bool:
        """Indica si el modelo soporta image editing."""
        return False
    
    def supports_image_variation(self) -> bool:
        """Indica si el modelo soporta variaciones de imagen."""
        return False
    
    def generate_image(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Image.Image]:
        """
        Genera imágenes desde texto (text-to-image).
        Override en modelos que soporten esta función.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} no soporta generación de imágenes desde texto"
        )
    
    def edit_image(
        self,
        image: Image.Image,
        prompt: str,
        mask: Optional[Image.Image] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Image.Image]:
        """
        Edita una imagen usando un prompt (image editing).
        Override en modelos que soporten esta función.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} no soporta edición de imágenes"
        )
    
    def create_variation(
        self,
        image: Image.Image,
        num_inference_steps: int = 50,
        strength: float = 0.75,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Image.Image]:
        """
        Crea variaciones de una imagen.
        Override en modelos que soporten esta función.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} no soporta variaciones de imagen"
        )


