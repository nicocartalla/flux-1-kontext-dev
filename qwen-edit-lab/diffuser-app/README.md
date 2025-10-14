# Diffuser API - OpenAI Compatible

API compatible con OpenAI para servir modelos diffusers de Hugging Face con FastAPI.

## 🎯 Características

- **Compatible con OpenAI API**: Implementa los endpoints estándar de OpenAI para imágenes
- **Modular y Extensible**: Arquitectura que permite definir todos los endpoints posibles
- **Control Total de Parámetros**: Cada modelo implementa solo los parámetros que necesita
- **Modelo Actual**: Qwen Image Edit (edición de imágenes)

## 📋 Estructura del Proyecto

```
diffuser-app/
├── src/
│   ├── api/              # Endpoints de FastAPI
│   │   ├── images.py     # Routers de imágenes (todos los endpoints)
│   │   └── dependencies.py
│   ├── config/           # Configuración
│   │   └── settings.py   # Settings y capacidades del modelo
│   ├── domain/           # Modelos de datos
│   │   └── openai_models.py  # Request/Response models (OpenAI compatible)
│   ├── model/            # Implementaciones de modelos
│   │   ├── base_model.py     # Clase base abstracta
│   │   └── qwen_image_edit.py # Implementación Qwen
│   ├── service/          # Lógica de negocio
│   │   └── image_service.py
│   └── main.py           # Aplicación FastAPI
├── requirements.txt
└── README.md
```

## 🚀 Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# Instalar versión de desarrollo de diffusers (recomendado)
pip install git+https://github.com/huggingface/diffusers
```

## 💻 Uso

### Iniciar el servidor

```bash
# Desde la carpeta diffuser-app
python -m src.main

# O con uvicorn directamente
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Variables de entorno (opcional)

```bash
# Configuración del modelo
export DIFFUSER_MODEL_ID="Qwen/Qwen-Image-Edit"
export DIFFUSER_DEVICE="cuda"  # cuda, cpu, mps
export DIFFUSER_TORCH_DTYPE="bfloat16"  # bfloat16, float16, float32

# Configuración del servidor
export DIFFUSER_HOST="0.0.0.0"
export DIFFUSER_PORT=8000

# Warmup (opcional)
export DIFFUSER_ENABLE_WARMUP=true
```

## 📡 Endpoints

### 1. POST /v1/images/edits (✅ Habilitado)

Edita una imagen usando un prompt de texto.

**Ejemplo con curl:**

```bash
curl -X POST "http://localhost:8000/v1/images/edits" \
  -F "image=@input.png" \
  -F "prompt=Change the rabbit's color to purple" \
  -F "num_inference_steps=50" \
  -F "true_cfg_scale=4.0" \
  -F "seed=42"
```

**Ejemplo con Python:**

```python
import requests
import base64
from PIL import Image
import io

# Cargar imagen
with open("input.png", "rb") as f:
    image_bytes = f.read()

# Hacer request
response = requests.post(
    "http://localhost:8000/v1/images/edits",
    files={"image": image_bytes},
    data={
        "prompt": "Change the rabbit's color to purple",
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
        "seed": 42,
        "response_format": "b64_json"
    }
)

# Obtener imagen
result = response.json()
img_data = base64.b64decode(result["data"][0]["b64_json"])
image = Image.open(io.BytesIO(img_data))
image.save("output.png")
```

**Parámetros:**

- `image` (required): Imagen a editar
- `prompt` (required): Instrucciones de edición
- `num_inference_steps` (opcional): Pasos de inferencia (default: 50)
- `true_cfg_scale` (opcional): CFG scale específico de Qwen (default: 4.0)
- `negative_prompt` (opcional): Prompt negativo (default: " ")
- `seed` (opcional): Seed para reproducibilidad
- `response_format` (opcional): "b64_json" o "url" (default: "b64_json")

### 2. POST /v1/images/generations (❌ No soportado)

Generación de imágenes desde texto. **No disponible** para Qwen Image Edit.

### 3. POST /v1/images/variations (❌ No soportado)

Variaciones de imagen. **No disponible** para Qwen Image Edit.

## 🏗️ Arquitectura

### Diseño Modular

La arquitectura está diseñada para:

1. **Definir todos los endpoints posibles** en `api/images.py`
2. **Cada modelo implementa solo lo que necesita** en `model/`
3. **Control total de parámetros** vía `domain/openai_models.py`
4. **Validación automática** mediante capabilities en `config/settings.py`

### Agregar un Nuevo Modelo

Para agregar un modelo de text-to-image (ej: Stable Diffusion):

1. **Crear implementación** en `src/model/stable_diffusion.py`:

```python
from .base_model import BaseDiffusionModel

class StableDiffusionModel(BaseDiffusionModel):
    def supports_text_to_image(self) -> bool:
        return True
    
    def generate_image(self, prompt: str, **kwargs):
        # Implementación
        pass
```

2. **Actualizar capabilities** en `src/config/settings.py`:

```python
class ModelCapabilities(BaseSettings):
    supports_text_to_image: bool = True  # ✅
    supports_image_edit: bool = False
```

3. **Usar en main.py**:

```python
model = StableDiffusionModel()
model.load_model()
```

## 🔧 Configuración Avanzada

### Capacidades del Modelo

Edita `src/config/settings.py` para cambiar las capacidades:

```python
class ModelCapabilities(BaseSettings):
    supports_text_to_image: bool = False
    supports_image_edit: bool = True      # Qwen soporta esto
    supports_image_variation: bool = False
    supports_inpainting: bool = False
    supports_mask: bool = False
```

### Parámetros del Modelo

Edita `src/config/settings.py` para ajustar defaults:

```python
class Settings(BaseSettings):
    default_num_inference_steps: int = 50
    default_true_cfg_scale: float = 4.0
    default_negative_prompt: str = " "
```

## 📚 API Compatible con OpenAI

Esta API implementa los endpoints estándar de OpenAI para imágenes:

- Mismos endpoints: `/v1/images/*`
- Mismos formatos de request/response
- Parámetros adicionales específicos de diffusers

Puedes usar el SDK de OpenAI (con URL personalizada):

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # No usamos API keys
)

# No soportado por OpenAI SDK para edits con archivos locales
# Usar requests directamente (ver ejemplo arriba)
```

## 🐳 Docker

Actualiza el Dockerfile existente:

```dockerfile
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

WORKDIR /app

# Copiar requirements
COPY diffuser-app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicación
COPY diffuser-app/src ./src

# Puerto
EXPOSE 8000

# Comando
CMD ["python", "-m", "src.main"]
```

## 📝 Testing

Ejemplo de test de imagen:

```python
# test_api.py
import requests

def test_image_edit():
    with open("input.png", "rb") as f:
        response = requests.post(
            "http://localhost:8000/v1/images/edits",
            files={"image": f},
            data={
                "prompt": "Make it purple",
                "seed": 42
            }
        )
    
    assert response.status_code == 200
    result = response.json()
    assert "data" in result
    assert len(result["data"]) > 0
```

## 📖 Documentación Interactiva

Una vez iniciado el servidor:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Info: http://localhost:8000/

## 🎯 Próximos Pasos

- [ ] Soporte para múltiples modelos simultáneos
- [ ] Almacenamiento de imágenes (S3, GCS) para `response_format=url`
- [ ] Rate limiting y autenticación
- [ ] Métricas y monitoring
- [ ] Batch processing
- [ ] WebSocket para streaming de generación

## 📄 Licencia

MIT

