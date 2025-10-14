# Dockerfile
#FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
FROM rayproject/ray:2.50.0-gpu
#rayproject/ray:2.37.0-gpu
# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar git para poder instalar diffusers desde GitHub
RUN sudo apt-get update && sudo apt-get install -y git && sudo rm -rf /var/lib/apt/lists/*

# 2. INSTALAMOS PYTORCH Y SUS DEPENDENCIAS DE CUDA
# La imagen base de Ray usa CUDA 12.1, por lo que instalamos la versión
# de PyTorch compatible.
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129



# Copiamos el archivo de requerimientos de la app
COPY diffuser-app/requirements.txt .

# Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Instalamos la versión de desarrollo de diffusers (recomendado)
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers

# Copiamos la aplicación
COPY diffuser-app/src ./src

# Copiamos la imagen de ejemplo (opcional, para testing)
COPY input.png ./input.png

# Puerto de la aplicación
EXPOSE 8000

# Variables de entorno por defecto
ENV DIFFUSER_HOST=0.0.0.0
ENV DIFFUSER_PORT=8000
ENV DIFFUSER_DEVICE=cuda
ENV DIFFUSER_TORCH_DTYPE=bfloat16

# Configurar PYTHONPATH para imports correctos
ENV PYTHONPATH=/app

# Comando que se ejecutará al iniciar el contenedor
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
