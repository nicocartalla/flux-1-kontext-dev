# 🏗️ Arquitectura - Diffuser API

Documentación de la arquitectura del sistema de serving de modelos diffusers con API compatible con OpenAI.

## 📐 Principios de Diseño

### 1. **Separación de Responsabilidades**
Cada capa tiene una responsabilidad específica y bien definida:
- **API Layer**: Maneja requests HTTP y validación
- **Service Layer**: Lógica de negocio y conversiones
- **Model Layer**: Implementación específica del modelo
- **Domain Layer**: Modelos de datos (requests/responses)
- **Config Layer**: Configuración y capabilities

### 2. **Extensibilidad**
La arquitectura permite:
- ✅ Definir **todos los endpoints posibles** de OpenAI
- ✅ Cada modelo implementa **solo lo que necesita**
- ✅ Agregar nuevos modelos sin modificar la API
- ✅ Control total de todos los parámetros

### 3. **OpenAI Compatibility**
- Endpoints compatibles: `/v1/images/*`
- Formatos de request/response idénticos
- Parámetros adicionales específicos de diffusers
- Puede usarse como drop-in replacement (con limitaciones)

## 📦 Estructura de Carpetas

```
diffuser-app/
│
├── src/
│   ├── __init__.py
│   │
│   ├── api/                      # 🌐 Capa de API (FastAPI)
│   │   ├── __init__.py
│   │   ├── images.py             # Todos los endpoints de imágenes
│   │   └── dependencies.py       # Dependency injection
│   │
│   ├── config/                   # ⚙️ Configuración
│   │   ├── __init__.py
│   │   └── settings.py           # Settings y ModelCapabilities
│   │
│   ├── domain/                   # 📋 Modelos de datos
│   │   ├── __init__.py
│   │   └── openai_models.py      # Request/Response (OpenAI spec)
│   │
│   ├── model/                    # 🤖 Implementaciones de modelos
│   │   ├── __init__.py
│   │   ├── base_model.py         # Clase base abstracta
│   │   └── qwen_image_edit.py    # Implementación Qwen
│   │
│   ├── service/                  # 🔧 Lógica de negocio
│   │   ├── __init__.py
│   │   └── image_service.py      # Conversiones y validaciones
│   │
│   └── main.py                   # 🚀 Aplicación principal
│
├── run.py                        # Script de entrada
├── requirements.txt              # Dependencias
├── test_client.py                # Cliente de prueba CLI
├── example_usage.py              # Ejemplos de uso
├── README.md                     # Documentación completa
├── QUICKSTART.md                 # Guía rápida
└── ARCHITECTURE.md               # Este archivo

```

## 🔄 Flujo de Datos

### Request Flow

```
Cliente HTTP
    │
    ├─> POST /v1/images/edits
    │
    ▼
┌─────────────────────────────────────────┐
│ API Layer (src/api/images.py)          │
│ - Validación de request                │
│ - Parsing de form-data                 │
│ - Verificación de capabilities         │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Service Layer (src/service/)            │
│ - Decodificación de imágenes           │
│ - Conversión de formatos                │
│ - Validaciones de negocio               │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Model Layer (src/model/)                │
│ - Llamada al pipeline de diffusers     │
│ - Manejo de GPU/CPU                     │
│ - Generación de imágenes                │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Response                                │
│ - Encoding a base64                     │
│ - Formato OpenAI standard               │
└─────────────────────────────────────────┘
```

### Startup Flow

```
main.py startup_event()
    │
    ├─> 1. Cargar configuración (settings.py)
    │
    ├─> 2. Instanciar modelo específico
    │      (QwenImageEditModel)
    │
    ├─> 3. model.load_model()
    │      - Descargar/cargar pesos
    │      - Mover a GPU/CPU
    │
    ├─> 4. model.warmup() (opcional)
    │      - Primera inferencia dummy
    │
    ├─> 5. Crear ImageService(model)
    │
    ├─> 6. set_image_service(service)
    │      - Inyectar en dependencies
    │
    └─> 7. Servidor listo ✅
```

## 🎯 Componentes Clave

### 1. API Layer (`src/api/images.py`)

**Responsabilidad**: Definir todos los endpoints posibles de OpenAI para imágenes.

```python
# Tres endpoints principales:
POST /v1/images/generations     # Text-to-Image
POST /v1/images/edits           # Image Editing ✅
POST /v1/images/variations      # Image Variation
```

**Características**:
- ✅ Define **todos** los parámetros posibles
- ✅ Valida capabilities antes de ejecutar
- ✅ Retorna errores 501 si no está soportado
- ✅ Manejo de multipart/form-data para imágenes

### 2. Domain Layer (`src/domain/openai_models.py`)

**Responsabilidad**: Modelos de datos (Pydantic) para requests y responses.

```python
class ImageEditRequest(BaseModel):
    # Parámetros OpenAI standard
    prompt: str
    image: str
    mask: Optional[str]
    
    # Parámetros adicionales diffusers
    num_inference_steps: Optional[int]
    true_cfg_scale: Optional[float]
    guidance_scale: Optional[float]
    negative_prompt: Optional[str]
    seed: Optional[int]
```

**Características**:
- ✅ Validación automática con Pydantic
- ✅ Documentación integrada (OpenAPI)
- ✅ Defaults configurables
- ✅ Enums para valores fijos

### 3. Model Layer (`src/model/`)

**Responsabilidad**: Implementaciones específicas de modelos.

#### Base Model (`base_model.py`)

Interfaz abstracta que define el contrato:

```python
class BaseDiffusionModel(ABC):
    @abstractmethod
    def load_model(self) -> None:
        pass
    
    def supports_text_to_image(self) -> bool:
        return False
    
    def supports_image_edit(self) -> bool:
        return False
    
    def generate_image(self, ...):
        raise NotImplementedError
    
    def edit_image(self, ...):
        raise NotImplementedError
```

#### Qwen Implementation (`qwen_image_edit.py`)

Implementación concreta para Qwen Image Edit:

```python
class QwenImageEditModel(BaseDiffusionModel):
    def supports_image_edit(self) -> bool:
        return True  # ✅
    
    def edit_image(self, image, prompt, **kwargs):
        # Implementación real
        with torch.inference_mode():
            output = self.pipeline(
                image=image,
                prompt=prompt,
                ...
            )
        return [output.images[0]]
```

**Características**:
- ✅ Solo implementa lo que necesita
- ✅ Maneja carga y descarga del modelo
- ✅ Warmup opcional
- ✅ Logging detallado

### 4. Service Layer (`src/service/image_service.py`)

**Responsabilidad**: Lógica de negocio entre API y modelo.

```python
class ImageService:
    def __init__(self, model: BaseDiffusionModel):
        self.model = model
    
    def decode_image(self, image_data: str) -> Image:
        # Decodifica base64 -> PIL Image
        
    def encode_image(self, image: Image) -> str:
        # Codifica PIL Image -> base64
        
    def edit_image(self, request, image_file) -> List[Image]:
        # Validaciones
        # Conversiones
        # Llamada al modelo
        return self.model.edit_image(...)
```

**Características**:
- ✅ Conversión de formatos (base64 ↔ PIL)
- ✅ Validaciones de negocio
- ✅ Gestión de errores
- ✅ Desacoplamiento API-Modelo

### 5. Config Layer (`src/config/settings.py`)

**Responsabilidad**: Configuración centralizada.

```python
class Settings(BaseSettings):
    # API Settings
    app_name: str = "Diffuser API"
    api_prefix: str = "/v1"
    
    # Model Settings
    model_id: str = "Qwen/Qwen-Image-Edit"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000

class ModelCapabilities(BaseSettings):
    supports_text_to_image: bool = False
    supports_image_edit: bool = True  ✅
    supports_image_variation: bool = False
```

**Características**:
- ✅ Variables de entorno con prefijo `DIFFUSER_`
- ✅ Defaults configurables
- ✅ Singleton pattern (`@lru_cache`)
- ✅ Capabilities para habilitar/deshabilitar endpoints

## 🔌 Dependency Injection

```python
# dependencies.py
image_service: Optional[ImageService] = None

def set_image_service(service: ImageService):
    global image_service
    image_service = service

def get_image_service() -> ImageService:
    return image_service

# main.py (startup)
model = QwenImageEditModel()
model.load_model()
service = ImageService(model)
set_image_service(service)

# api/images.py (endpoint)
async def create_image_edit(
    service: ImageService = Depends(get_image_service)
):
    ...
```

## 🎨 Agregar un Nuevo Modelo

### Ejemplo: Stable Diffusion (Text-to-Image)

#### 1. Crear implementación

```python
# src/model/stable_diffusion.py
from .base_model import BaseDiffusionModel
from diffusers import StableDiffusionPipeline

class StableDiffusionModel(BaseDiffusionModel):
    def load_model(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2"
        )
        self.pipeline.to("cuda")
    
    def supports_text_to_image(self) -> bool:
        return True
    
    def generate_image(self, prompt, **kwargs):
        with torch.inference_mode():
            images = self.pipeline(prompt, **kwargs).images
        return images
```

#### 2. Actualizar capabilities

```python
# src/config/settings.py
class ModelCapabilities(BaseSettings):
    supports_text_to_image: bool = True  # ✅
    supports_image_edit: bool = False
```

#### 3. Usar en main.py

```python
# src/main.py
from .model.stable_diffusion import StableDiffusionModel

@app.on_event("startup")
async def startup_event():
    model = StableDiffusionModel()  # Cambio aquí
    model.load_model()
    service = ImageService(model)
    set_image_service(service)
```

#### 4. ¡Listo! ✅

El endpoint `/v1/images/generations` ahora está habilitado automáticamente.

## 🔐 Validación de Capabilities

```python
# api/images.py
@router.post("/edits")
async def create_image_edit(...):
    capabilities = get_model_capabilities()
    
    if not capabilities.supports_image_edit:
        raise HTTPException(
            status_code=501,
            detail={
                "error": {
                    "message": "Modelo no soporta edición",
                    "type": "not_supported_error"
                }
            }
        )
    
    # Procesar...
```

**Beneficios**:
- ✅ Endpoints se habilitan/deshabilitan automáticamente
- ✅ Errores claros si se intenta usar algo no soportado
- ✅ Documentación actualizada en `/docs`

## 📊 Logging y Observabilidad

```python
# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# En cada componente
logger = logging.getLogger(__name__)

# Ejemplos
logger.info("Modelo cargado exitosamente")
logger.error("Error editando imagen", exc_info=True)
```

**Health Checks**:
```bash
GET /health        # Status del servicio
GET /              # Info completa + capabilities
```

## 🚀 Despliegue

### Variables de Entorno Críticas

```bash
DIFFUSER_DEVICE=cuda              # GPU
DIFFUSER_TORCH_DTYPE=bfloat16     # Precisión
DIFFUSER_MODEL_ID=Qwen/Qwen-Image-Edit
```

### Resources

```yaml
# Kubernetes
resources:
  limits:
    nvidia.com/gpu: 1
    memory: "32Gi"
```

## 🎯 Ventajas de esta Arquitectura

| Ventaja | Descripción |
|---------|-------------|
| **Extensible** | Agregar modelos sin tocar la API |
| **Modular** | Cada capa independiente |
| **OpenAI Compatible** | Drop-in replacement (parcial) |
| **Type Safe** | Pydantic para validación |
| **Documentado** | OpenAPI/Swagger automático |
| **Observable** | Logging estructurado |
| **Configurable** | Variables de entorno |
| **Testeable** | Dependency injection |

## 🔮 Futuras Mejoras

- [ ] Múltiples modelos simultáneos (model routing)
- [ ] Almacenamiento de imágenes (S3/GCS)
- [ ] Rate limiting y autenticación
- [ ] Métricas Prometheus
- [ ] Batch processing
- [ ] Async queue para long-running tasks
- [ ] WebSocket streaming
- [ ] Model caching y warm pools

## 📚 Referencias

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/images)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

---

**Diseño basado en**:
- Clean Architecture
- Dependency Inversion Principle
- OpenAI API Standards
- Ray Serve patterns (de los ejemplos stable_diffusion.py)

