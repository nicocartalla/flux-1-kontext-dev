"""
Ejemplos de uso de la API de Diffuser.
"""
import requests
import base64
from PIL import Image
import io


def example_basic_edit():
    """Ejemplo básico de edición de imagen."""
    print("=" * 80)
    print("Ejemplo 1: Edición básica")
    print("=" * 80)
    
    # Cargar imagen
    with open("input.png", "rb") as f:
        image_bytes = f.read()
    
    # Request
    response = requests.post(
        "http://localhost:8000/v1/images/edits",
        files={"image": ("image.png", image_bytes, "image/png")},
        data={
            "prompt": "Change the rabbit's color to purple",
            "response_format": "b64_json"
        }
    )
    
    # Guardar resultado
    if response.status_code == 200:
        result = response.json()
        img_data = base64.b64decode(result["data"][0]["b64_json"])
        image = Image.open(io.BytesIO(img_data))
        image.save("example1_output.png")
        print("✅ Imagen guardada en: example1_output.png\n")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())


def example_with_parameters():
    """Ejemplo con parámetros personalizados."""
    print("=" * 80)
    print("Ejemplo 2: Con parámetros personalizados")
    print("=" * 80)
    
    with open("input.png", "rb") as f:
        image_bytes = f.read()
    
    response = requests.post(
        "http://localhost:8000/v1/images/edits",
        files={"image": ("image.png", image_bytes, "image/png")},
        data={
            "prompt": "Add a flashlight background and make it neon green",
            "num_inference_steps": 100,  # Más pasos = mejor calidad
            "true_cfg_scale": 5.0,       # Mayor CFG = más fidelidad al prompt
            "seed": 42,                  # Para reproducibilidad
            "negative_prompt": "blurry, distorted, low quality",
            "response_format": "b64_json"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        img_data = base64.b64decode(result["data"][0]["b64_json"])
        image = Image.open(io.BytesIO(img_data))
        image.save("example2_output.png")
        print("✅ Imagen guardada en: example2_output.png")
        print(f"   Timestamp: {result['created']}\n")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())


def example_multiple_variations():
    """Ejemplo generando múltiples variaciones con diferentes seeds."""
    print("=" * 80)
    print("Ejemplo 3: Múltiples variaciones (diferentes seeds)")
    print("=" * 80)
    
    with open("input.png", "rb") as f:
        image_bytes = f.read()
    
    prompt = "Make the background a starry night sky"
    
    for i, seed in enumerate([42, 123, 456, 789]):
        print(f"  Generando variación {i+1} con seed={seed}...")
        
        response = requests.post(
            "http://localhost:8000/v1/images/edits",
            files={"image": ("image.png", image_bytes, "image/png")},
            data={
                "prompt": prompt,
                "seed": seed,
                "num_inference_steps": 50,
                "true_cfg_scale": 4.0,
                "response_format": "b64_json"
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            img_data = base64.b64decode(result["data"][0]["b64_json"])
            image = Image.open(io.BytesIO(img_data))
            image.save(f"example3_output_{seed}.png")
            print(f"  ✅ Guardado: example3_output_{seed}.png")
        else:
            print(f"  ❌ Error: {response.status_code}")
    
    print()


def example_check_capabilities():
    """Ejemplo para verificar las capacidades del modelo."""
    print("=" * 80)
    print("Ejemplo 4: Verificar capacidades del modelo")
    print("=" * 80)
    
    response = requests.get("http://localhost:8000/")
    
    if response.status_code == 200:
        info = response.json()
        print(f"Servicio: {info['service']}")
        print(f"Versión: {info['version']}")
        print(f"Modelo: {info['model']}")
        print("\nCapacidades:")
        for cap, enabled in info['capabilities'].items():
            icon = "✅" if enabled else "❌"
            print(f"  {icon} {cap}")
        print("\nEndpoints disponibles:")
        for name, path in info['endpoints'].items():
            print(f"  - {name}: {path}")
    else:
        print(f"❌ Error: {response.status_code}")
    
    print()


def example_error_handling():
    """Ejemplo de manejo de errores."""
    print("=" * 80)
    print("Ejemplo 5: Manejo de errores")
    print("=" * 80)
    
    # Intentar usar un endpoint no soportado
    print("Intentando generar imagen desde texto (no soportado)...")
    response = requests.post(
        "http://localhost:8000/v1/images/generations",
        json={
            "prompt": "A beautiful sunset",
            "size": "1024x1024"
        }
    )
    
    if response.status_code != 200:
        print(f"❌ Status: {response.status_code}")
        error = response.json()
        print(f"   Error: {error.get('detail', {}).get('error', {}).get('message', 'Unknown')}")
    
    print()


if __name__ == "__main__":
    print("\n🎨 Ejemplos de uso de Diffuser API\n")
    
    # Verificar que el servidor esté corriendo
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code != 200:
            print("❌ El servidor no está respondiendo correctamente")
            print("   Asegúrate de que esté corriendo: python -m src.main")
            exit(1)
    except requests.exceptions.RequestException:
        print("❌ No se pudo conectar al servidor")
        print("   Asegúrate de que esté corriendo en http://localhost:8000")
        print("   Ejecuta: python -m src.main")
        exit(1)
    
    print("✅ Servidor conectado\n")
    
    # Ejecutar ejemplos
    try:
        example_check_capabilities()
        example_basic_edit()
        example_with_parameters()
        example_multiple_variations()
        example_error_handling()
        
        print("=" * 80)
        print("✅ Todos los ejemplos completados")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error ejecutando ejemplos: {e}")
        import traceback
        traceback.print_exc()

