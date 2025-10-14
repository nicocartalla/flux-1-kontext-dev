"""
Implementación del modelo Qwen Image Edit.
Este modelo solo soporta edición de imágenes (image editing).
"""
from typing import List, Optional
from PIL import Image
import torch
import logging
from diffusers import QwenImageEditPipeline

from .base_model import BaseDiffusionModel
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class QwenImageEditModel(BaseDiffusionModel):
    """
    Modelo Qwen Image Edit para edición de imágenes.
    Solo implementa el método edit_image.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.pipeline = None
        self.device = self.settings.device
        
    def load_model(self) -> None:
        """Carga el pipeline de Qwen Image Edit."""
        logger.info(f"Cargando modelo {self.settings.model_id}...")
        
        # Determinar el dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.settings.torch_dtype, torch.bfloat16)
        
        # Cargar el pipeline
        self.pipeline = QwenImageEditPipeline.from_pretrained(
            self.settings.model_id,
            torch_dtype=torch_dtype
        )
        
        # Mover a dispositivo
        self.pipeline.to(self.device)
        
        logger.info(f"Modelo cargado exitosamente en {self.device}")
    
    def warmup(self) -> None:
        """Realiza un warmup opcional del modelo."""
        if not self.settings.enable_warmup:
            return
            
        logger.info("Realizando warmup del modelo...")
        
        # Crear una imagen dummy
        dummy_image = Image.new("RGB", (512, 512), color="white")
        dummy_prompt = "warmup request"
        
        try:
            with torch.inference_mode():
                _ = self.pipeline(
                    image=dummy_image,
                    prompt=dummy_prompt,
                    generator=torch.manual_seed(0),
                    num_inference_steps=10,  # Pocas steps para warmup
                )
            logger.info("Warmup completado exitosamente")
        except Exception as e:
            logger.warning(f"Error durante warmup: {e}")
    
    def supports_image_edit(self) -> bool:
        """Qwen Image Edit soporta edición de imágenes."""
        return True
    
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
        Edita una imagen usando Qwen Image Edit.
        
        Args:
            image: Imagen PIL a editar
            prompt: Instrucciones de edición
            mask: No usado en Qwen (pero mantenido por compatibilidad)
            num_inference_steps: Pasos de inferencia
            guidance_scale: No usado en Qwen directamente
            negative_prompt: Prompt negativo
            seed: Seed para reproducibilidad
            **kwargs: Parámetros adicionales (true_cfg_scale, etc.)
            
        Returns:
            Lista con la imagen editada
        """
        if self.pipeline is None:
            raise RuntimeError("Modelo no cargado. Llama a load_model() primero.")
        
        # Convertir a RGB si es necesario
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Preparar parámetros
        true_cfg_scale = kwargs.get(
            "true_cfg_scale", 
            self.settings.default_true_cfg_scale
        )
        
        if negative_prompt is None:
            negative_prompt = self.settings.default_negative_prompt
        
        # Preparar generator si hay seed
        generator = None
        if seed is not None:
            generator = torch.manual_seed(seed)
        
        # Construir inputs
        inputs = {
            "image": image,
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt,
        }
        
        if generator is not None:
            inputs["generator"] = generator
        
        logger.info(
            f"Editando imagen - Steps: {num_inference_steps}, "
            f"CFG: {true_cfg_scale}, Seed: {seed}"
        )
        
        # Ejecutar inferencia
        with torch.inference_mode():
            output = self.pipeline(**inputs)
            result_image = output.images[0]
        
        logger.info("Edición completada exitosamente")
        
        return [result_image]

