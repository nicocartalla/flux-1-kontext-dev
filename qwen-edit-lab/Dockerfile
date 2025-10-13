# Dockerfile
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime


# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos el archivo de requerimientos
COPY requirements.txt .

#
RUN pip install fastapi

# Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el script y la imagen de entrada al contenedor
COPY test_model.py .
COPY input.png .

# Comando que se ejecutará al iniciar el contenedor
CMD ["python", "test_model.py"]
