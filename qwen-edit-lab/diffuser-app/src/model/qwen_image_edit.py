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
        """Carga el pipeline de Qwen Image Edit con optimizaciones para H200."""
        logger.info(f"Cargando modelo {self.settings.model_id}...")
        
        # ====================================================================
        # 1. Información de GPU/CUDA
        # ====================================================================
        if torch.cuda.is_available():
            logger.info(f"🚀 CUDA disponible: {torch.cuda.get_device_name(0)}")
            logger.info(f"🚀 Versión CUDA: {torch.version.cuda}")
            logger.info(f"🚀 Número de GPUs: {torch.cuda.device_count()}")
            logger.info(f"🚀 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            compute_cap = torch.cuda.get_device_capability(0)
            logger.info(f"🚀 Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
        else:
            logger.warning("⚠️ CUDA NO está disponible, usando CPU")
        
        # ====================================================================
        # 2. Habilitar optimizaciones de cuDNN [CRÍTICO]
        # ====================================================================
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("✅ cuDNN benchmark habilitado")
            logger.info("✅ TensorFloat-32 (TF32) habilitado")
        
        # ====================================================================
        # 3. Determinar el dtype
        # ====================================================================
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.settings.torch_dtype, torch.bfloat16)
        logger.info(f"🔧 Usando dtype: {torch_dtype}")
        
        # ====================================================================
        # 4. Cargar el pipeline
        # ====================================================================
        self.pipeline = QwenImageEditPipeline.from_pretrained(
            self.settings.model_id,
            torch_dtype=torch_dtype
        )
        
        # ====================================================================
        # 5. Mover a dispositivo
        # ====================================================================
        self.pipeline.to(self.device)
        
        # ====================================================================
        # 6. Aplicar optimizaciones de memoria y velocidad
        # ====================================================================
        if self.device == "cuda":
            try:
                # Intentar habilitar xFormers (memory-efficient attention)
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("✅ xFormers memory-efficient attention habilitado")
            except Exception as e:
                logger.warning(f"⚠️ xFormers no disponible: {e}")
                try:
                    # Fallback: attention slicing
                    self.pipeline.enable_attention_slicing()
                    logger.info("✅ Attention slicing habilitado (fallback)")
                except Exception as e2:
                    logger.warning(f"⚠️ No se pudo habilitar attention slicing: {e2}")
            
            # Detectar y optimizar el componente principal del pipeline
            # Diferentes pipelines usan diferentes nombres: unet, transformer, dit, etc.
            pytorch_version = tuple(map(int, torch.__version__.split('.')[:2]))
            if pytorch_version >= (2, 0):
                # Buscar el componente principal
                main_component = None
                component_name = None
                
                for name in ['transformer', 'dit', 'unet', 'model']:
                    if hasattr(self.pipeline, name):
                        comp = getattr(self.pipeline, name)
                        if comp is not None and hasattr(comp, 'parameters'):
                            main_component = comp
                            component_name = name
                            logger.info(f"🎯 Componente principal detectado: {component_name}")
                            break
                
                if main_component is not None:
                    try:
                        logger.info(f"🔥 Compilando {component_name} con torch.compile()...")
                        compiled_component = torch.compile(
                            main_component,
                            mode="reduce-overhead",
                            fullgraph=False
                        )
                        setattr(self.pipeline, component_name, compiled_component)
                        logger.info(f"✅ {component_name} compilado con torch.compile() [modo: reduce-overhead]")
                        
                        # Optimizar layout de memoria para Tensor Cores
                        try:
                            compiled_component.to(memory_format=torch.channels_last)
                            logger.info(f"✅ {component_name} configurado con channels_last")
                        except Exception as e:
                            logger.warning(f"⚠️ No se pudo configurar channels_last en {component_name}: {e}")
                            
                    except Exception as e:
                        logger.warning(f"⚠️ No se pudo compilar {component_name}: {e}")
                else:
                    logger.warning("⚠️ No se encontró componente principal para optimizar")
            else:
                logger.warning(f"⚠️ PyTorch {torch.__version__} no soporta torch.compile()")
        
        # ====================================================================
        # 7. Verificar que el modelo está en GPU
        # ====================================================================
        if self.device == "cuda" and torch.cuda.is_available():
            # Verificar dispositivo de componentes principales
            for comp_name in ['transformer', 'text_encoder', 'vae']:
                try:
                    comp = getattr(self.pipeline, comp_name, None)
                    if comp is not None and hasattr(comp, 'parameters'):
                        device = next(comp.parameters()).device
                        logger.info(f"✅ {comp_name} en dispositivo: {device}")
                except Exception as e:
                    logger.debug(f"No se pudo verificar {comp_name}: {e}")
            
            logger.info(f"✅ Modelo cargado exitosamente en GPU: {self.device}")
            logger.info(f"✅ GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            logger.info(f"✅ GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        else:
            logger.info(f"ℹ️ Modelo cargado en: {self.device}")
    
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


