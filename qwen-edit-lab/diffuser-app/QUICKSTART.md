# 🚀 Quickstart - Diffuser API

Guía rápida para poner en marcha el servidor de edición de imágenes con Qwen.

## 📦 Instalación Local

### 1. Clonar e instalar dependencias

```bash
# Navegar a la carpeta
cd local-tests/flux-1-kontext-dev/qwen-edit-lab/diffuser-app

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar diffusers de desarrollo (recomendado)
pip install git+https://github.com/huggingface/diffusers
```

### 2. Iniciar el servidor

```bash
# Opción 1: Usando el módulo main
python -m src.main

# Opción 2: Usando el script run.py
python run.py

# Opción 3: Con uvicorn directamente
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 3. Verificar que funcione

```bash
# En otra terminal
curl http://localhost:8000/health
# Debería retornar: {"status":"healthy"}

# Ver información del servicio
curl http://localhost:8000/
```

### 4. Probar la edición de imágenes

**Opción A: Con el script de prueba**

```bash
# Verificar salud
python test_client.py health

# Ver info
python test_client.py info

# Editar imagen
python test_client.py edit \
  --image ../input.png \
  --prompt "Change the rabbit's color to purple" \
  --output output_purple.png \
  --steps 50 \
  --cfg 4.0 \
  --seed 42
```

**Opción B: Con curl**

```bash
curl -X POST "http://localhost:8000/v1/images/edits" \
  -F "image=@../input.png" \
  -F "prompt=Change the rabbit's color to purple" \
  -F "num_inference_steps=50" \
  -F "true_cfg_scale=4.0" \
  -F "seed=42" \
  -o response.json
```

**Opción C: Con Python**

```python
import requests
import base64
from PIL import Image
import io

# Cargar imagen
with open("../input.png", "rb") as f:
    image_bytes = f.read()

# Request
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

# Guardar
result = response.json()
img_data = base64.b64decode(result["data"][0]["b64_json"])
image = Image.open(io.BytesIO(img_data))
image.save("output.png")
```

**Opción D: Con los ejemplos incluidos**

```bash
python example_usage.py
# Esto ejecutará varios ejemplos automáticamente
```

## 🐳 Docker

### Build

```bash
# Desde la carpeta qwen-edit-lab (padre de diffuser-app)
cd local-tests/flux-1-kontext-dev/qwen-edit-lab

docker build -t qwen-image-edit:latest .
```

### Run

```bash
# Básico
docker run --gpus all -p 8000:8000 qwen-image-edit:latest

# Con variables de entorno
docker run --gpus all -p 8000:8000 \
  -e DIFFUSER_ENABLE_WARMUP=true \
  -e DIFFUSER_DEFAULT_NUM_INFERENCE_STEPS=100 \
  qwen-image-edit:latest

# Con volumen para persistir caché
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  qwen-image-edit:latest
```

### Push (ejemplo para GCP Artifact Registry)

```bash
# Tag
docker tag qwen-image-edit:latest \
  us-east4-docker.pkg.dev/genai-aas/test-images/qwen-image:lab3

# Push
docker push us-east4-docker.pkg.dev/genai-aas/test-images/qwen-image:lab3
```

## ☸️ Kubernetes

### Deploy en GKE

```bash
# Aplicar pod
kubectl apply -f demo-pod.yaml

# Ver logs
kubectl logs demo-qwen-lab -f

# Crear servicio (opcional)
kubectl apply -f k8s-service.yaml

# Port forward para testing
kubectl port-forward demo-qwen-lab 8000:8000

# Probar
curl http://localhost:8000/health
```

### Build en GKE con Buildah

Si usas el sistema de buildah del repo:

```bash
# Actualizar el build-images.yaml con la nueva estructura
# Luego aplicar
kubectl apply -f ../../build-images.yaml

# Ver progreso
kubectl logs -f job/buildah-privileged-job
```

## 📊 Monitoreo

### Logs

```bash
# Docker
docker logs <container-id> -f

# Kubernetes
kubectl logs demo-qwen-lab -f

# Local
# Los logs aparecen directamente en la terminal
```

### Health Checks

```bash
# Health
curl http://localhost:8000/health

# Info completa
curl http://localhost:8000/ | jq

# Documentación
open http://localhost:8000/docs
```

## ⚙️ Configuración

### Variables de Entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `DIFFUSER_MODEL_ID` | `Qwen/Qwen-Image-Edit` | ID del modelo en HuggingFace |
| `DIFFUSER_DEVICE` | `cuda` | Dispositivo: cuda, cpu, mps |
| `DIFFUSER_TORCH_DTYPE` | `bfloat16` | Tipo de datos: bfloat16, float16, float32 |
| `DIFFUSER_HOST` | `0.0.0.0` | Host del servidor |
| `DIFFUSER_PORT` | `8000` | Puerto del servidor |
| `DIFFUSER_ENABLE_WARMUP` | `false` | Habilitar warmup al inicio |
| `DIFFUSER_DEFAULT_NUM_INFERENCE_STEPS` | `50` | Steps por defecto |
| `DIFFUSER_DEFAULT_TRUE_CFG_SCALE` | `4.0` | CFG scale por defecto |

### Ejemplo de archivo .env

Crea un archivo `.env` en `diffuser-app/`:

```bash
DIFFUSER_DEVICE=cuda
DIFFUSER_TORCH_DTYPE=bfloat16
DIFFUSER_ENABLE_WARMUP=true
DIFFUSER_DEFAULT_NUM_INFERENCE_STEPS=75
```

Luego carga con:

```bash
export $(cat .env | xargs)
python -m src.main
```

## 🔧 Troubleshooting

### Error: "CUDA out of memory"

```bash
# Reducir steps
export DIFFUSER_DEFAULT_NUM_INFERENCE_STEPS=25

# Usar float16 en lugar de bfloat16
export DIFFUSER_TORCH_DTYPE=float16
```

### Error: "Model not loaded"

```bash
# Verificar que se haya descargado el modelo
ls ~/.cache/huggingface/hub/

# Descargar manualmente
python -c "from diffusers import QwenImageEditPipeline; QwenImageEditPipeline.from_pretrained('Qwen/Qwen-Image-Edit')"
```

### Puerto ocupado

```bash
# Cambiar puerto
export DIFFUSER_PORT=8001
python -m src.main
```

### Verificar GPU

```bash
# Verificar CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Ver info de GPU
nvidia-smi
```

## 📚 Más Información

- [README completo](README.md)
- [Documentación interactiva](http://localhost:8000/docs) (después de iniciar el servidor)
- [Ejemplos de uso](example_usage.py)
- [Cliente de prueba](test_client.py)

## 🎯 Próximos Pasos

1. ✅ Servidor funcionando
2. 🔄 Prueba con diferentes prompts
3. 🎨 Experimenta con parámetros (steps, cfg_scale, seed)
4. 📦 Empaqueta con Docker
5. ☸️ Despliega en Kubernetes
6. 🚀 Integra con tu aplicación

---

**¿Problemas?** Revisa los logs y el [README](README.md) para más detalles.

