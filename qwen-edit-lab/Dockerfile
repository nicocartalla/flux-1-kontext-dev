# Dockerfile
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

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

# Comando que se ejecutará al iniciar el contenedor
CMD ["python", "-m", "src.main"]
