"""
Script de prueba para el API de edición de imágenes.
"""
import requests
import base64
from PIL import Image
import io
import argparse


def test_image_edit(
    image_path: str,
    prompt: str,
    output_path: str = "output_edited.png",
    api_url: str = "http://localhost:8000",
    num_inference_steps: int = 50,
    true_cfg_scale: float = 4.0,
    seed: int = None
):
    """
    Prueba el endpoint de edición de imágenes.
    
    Args:
        image_path: Ruta a la imagen de entrada
        prompt: Prompt de edición
        output_path: Ruta para guardar la imagen editada
        api_url: URL base del API
        num_inference_steps: Pasos de inferencia
        true_cfg_scale: CFG scale de Qwen
        seed: Seed para reproducibilidad
    """
    print(f"Cargando imagen desde: {image_path}")
    
    # Leer imagen
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # Preparar datos del form
    data = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "response_format": "b64_json"
    }
    
    if seed is not None:
        data["seed"] = seed
    
    print(f"\nEnviando request al API: {api_url}/v1/images/edits")
    print(f"Prompt: {prompt}")
    print(f"Steps: {num_inference_steps}, CFG: {true_cfg_scale}, Seed: {seed}")
    
    # Hacer request
    response = requests.post(
        f"{api_url}/v1/images/edits",
        files={"image": ("image.png", image_bytes, "image/png")},
        data=data
    )
    
    # Verificar respuesta
    if response.status_code != 200:
        print(f"\n❌ Error: {response.status_code}")
        print(response.json())
        return
    
    print("✅ Request exitoso!")
    
    # Obtener imagen
    result = response.json()
    img_data = base64.b64decode(result["data"][0]["b64_json"])
    image = Image.open(io.BytesIO(img_data))
    
    # Guardar
    image.save(output_path)
    print(f"\n💾 Imagen guardada en: {output_path}")
    print(f"📏 Tamaño: {image.size}")
    print(f"🕐 Timestamp: {result['created']}")


def test_health(api_url: str = "http://localhost:8000"):
    """Prueba el health check."""
    print(f"Probando health check: {api_url}/health")
    response = requests.get(f"{api_url}/health")
    
    if response.status_code == 200:
        print("✅ API está saludable")
        print(response.json())
    else:
        print(f"❌ Error: {response.status_code}")


def test_info(api_url: str = "http://localhost:8000"):
    """Obtiene información del servicio."""
    print(f"Obteniendo info del servicio: {api_url}/")
    response = requests.get(f"{api_url}/")
    
    if response.status_code == 200:
        info = response.json()
        print("\n📋 Información del servicio:")
        print(f"  Servicio: {info['service']}")
        print(f"  Versión: {info['version']}")
        print(f"  Modelo: {info['model']}")
        print(f"\n🎯 Capacidades:")
        for cap, enabled in info['capabilities'].items():
            icon = "✅" if enabled else "❌"
            print(f"  {icon} {cap}: {enabled}")
        print(f"\n🔗 Endpoints:")
        for name, path in info['endpoints'].items():
            print(f"  - {name}: {path}")
    else:
        print(f"❌ Error: {response.status_code}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cliente de prueba para Diffuser API")
    parser.add_argument(
        "command",
        choices=["health", "info", "edit"],
        help="Comando a ejecutar"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Ruta a la imagen de entrada (para edit)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt de edición (para edit)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_edited.png",
        help="Ruta de salida (para edit)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="URL del API"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Número de pasos de inferencia"
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=4.0,
        help="True CFG scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed para reproducibilidad"
    )
    
    args = parser.parse_args()
    
    if args.command == "health":
        test_health(args.api_url)
    
    elif args.command == "info":
        test_info(args.api_url)
    
    elif args.command == "edit":
        if not args.image or not args.prompt:
            print("❌ Error: --image y --prompt son requeridos para 'edit'")
            parser.print_help()
        else:
            test_image_edit(
                image_path=args.image,
                prompt=args.prompt,
                output_path=args.output,
                api_url=args.api_url,
                num_inference_steps=args.steps,
                true_cfg_scale=args.cfg,
                seed=args.seed
            )

