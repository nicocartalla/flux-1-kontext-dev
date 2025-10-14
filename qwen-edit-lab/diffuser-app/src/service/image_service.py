"""
Servicio de lógica de negocio para operaciones de imagen.
Maneja la conversión entre formatos, validaciones, etc.
"""
from typing import List, Optional
from PIL import Image
import io
import base64
import logging

from ..model.base_model import BaseDiffusionModel
from ..domain.openai_models import (
    ImageEditRequest,
    ImageGenerationRequest,
    ImageVariationRequest,
    ResponseFormat
)

logger = logging.getLogger(__name__)


class ImageService:
    """
    Servicio para operaciones de imagen.
    Actúa como capa intermedia entre la API y el modelo.
    """
    
    def __init__(self, model: BaseDiffusionModel):
        self.model = model
    
    @staticmethod
    def decode_image(image_data: str) -> Image.Image:
        """
        Decodifica una imagen desde base64 o bytes.
        
        Args:
            image_data: String en base64 o bytes
            
        Returns:
            Imagen PIL
        """
        try:
            # Intentar decodificar desde base64
            if isinstance(image_data, str):
                # Remover prefijo data:image/... si existe
                if "base64," in image_data:
                    image_data = image_data.split("base64,")[1]
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data
            
            image = Image.open(io.BytesIO(image_bytes))
            return image.convert("RGB")
            
        except Exception as e:
            logger.error(f"Error decodificando imagen: {e}")
            raise ValueError(f"No se pudo decodificar la imagen: {e}")
    
    @staticmethod
    def encode_image(image: Image.Image, format: ResponseFormat = ResponseFormat.B64_JSON) -> str:
        """
        Codifica una imagen PIL a base64.
        
        Args:
            image: Imagen PIL
            format: Formato de respuesta
            
        Returns:
            String en base64
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def edit_image(
        self,
        request: ImageEditRequest,
        image_file: Optional[bytes] = None
    ) -> List[Image.Image]:
        """
        Edita una imagen según el request.
        
        Args:
            request: Request de edición
            image_file: Bytes de la imagen (desde form-data)
            
        Returns:
            Lista de imágenes editadas
        """
        # Validar que el modelo soporta edición
        if not self.model.supports_image_edit():
            raise NotImplementedError(
                "El modelo cargado no soporta edición de imágenes"
            )
        
        # Decodificar imagen
        if image_file:
            image = self.decode_image(image_file)
        else:
            image = self.decode_image(request.image)
        
        # Decodificar máscara si existe
        mask = None
        if request.mask:
            mask = self.decode_image(request.mask)
        
        # Llamar al modelo
        images = self.model.edit_image(
            image=image,
            prompt=request.prompt,
            mask=mask,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            true_cfg_scale=request.true_cfg_scale,
        )
        
        return images
    
    def generate_image(
        self,
        request: ImageGenerationRequest
    ) -> List[Image.Image]:
        """
        Genera imágenes desde texto.
        
        Args:
            request: Request de generación
            
        Returns:
            Lista de imágenes generadas
        """
        if not self.model.supports_text_to_image():
            raise NotImplementedError(
                "El modelo cargado no soporta generación de imágenes desde texto"
            )
        
        images = self.model.generate_image(
            prompt=request.prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
        )
        
        return images
    
    def create_variation(
        self,
        request: ImageVariationRequest,
        image_file: Optional[bytes] = None
    ) -> List[Image.Image]:
        """
        Crea variaciones de una imagen.
        
        Args:
            request: Request de variación
            image_file: Bytes de la imagen
            
        Returns:
            Lista de variaciones
        """
        if not self.model.supports_image_variation():
            raise NotImplementedError(
                "El modelo cargado no soporta variaciones de imagen"
            )
        
        # Decodificar imagen
        if image_file:
            image = self.decode_image(image_file)
        else:
            image = self.decode_image(request.image)
        
        images = self.model.create_variation(
            image=image,
            num_inference_steps=request.num_inference_steps,
            strength=request.strength,
            seed=request.seed,
        )
        
        return images

