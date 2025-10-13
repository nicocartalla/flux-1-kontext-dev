# test_model.py
import os
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline

print("Iniciando el script de edición...")

# Carga el modelo (esto puede tardar la primera vez)
pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.bfloat16
)
print("Pipeline del modelo cargada.")

# Mueve el pipeline a la GPU
pipeline.to("cuda")
print("Modelo movido a la GPU (cuda).")

# Carga la imagen de entrada desde la misma carpeta
image_path = "./input.png"
image = Image.open(image_path).convert("RGB")
print(f"Imagen de entrada '{image_path}' cargada.")

prompt = "Change the rabbit's color to purple, with a flash light background."

inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

print("Ejecutando inferencia... Esto puede tardar unos minutos.")
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit.png")

print("¡Inferencia completada!")
print(f"Imagen guardada en: /app/output_image_edit.png (dentro del contenedor)")
