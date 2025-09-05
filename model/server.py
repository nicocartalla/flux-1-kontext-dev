# server-full.py (v1.3.1; fix cloudpickle + H100 tuning)
from io import BytesIO
from typing import Optional, List, Any
import os
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, HttpUrl
from PIL import Image

from ray import serve
from ray.serve.handle import DeploymentHandle


# =========================== Helpers de ENV ===========================
def _flag(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in ("1", "true", "yes")

HF_TOKEN_ENV = (os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN") or "").strip()
HF_CACHE = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or "/tmp/hf-cache"

# NO import de torch en nivel módulo. Evita 'cannot pickle CudnnModule'.

USE_DEVICE_MAP = _flag("USE_DEVICE_MAP", "0")
USE_FLASH = _flag("USE_FLASH", "1")          # activa SDPA/Flash por defecto
USE_COMPILE = _flag("USE_COMPILE", "1")      # compilar el bloque principal por defecto

# Mantén estos desde ENV para no depender de torch al importar el módulo.
ACTOR_GPUS = max(1, int(os.getenv("ACTOR_GPUS", str(2 if USE_DEVICE_MAP else 1))))
REPLICAS = max(1, int(os.getenv("REPLICAS", "1")))

BATCH_MAX_SIZE = int(os.getenv("BATCH_MAX_SIZE", "4"))
BATCH_WAIT_MS = int(os.getenv("BATCH_WAIT_MS", "10"))  # ms

# Recom. memoria CUDA (menos fragmentación)
DEFAULT_ALLOC_CONF = "expandable_segments:True"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", DEFAULT_ALLOC_CONF)

# =========================== FastAPI ===========================
app = FastAPI(title="FluxKontext Server (H100-optimized)", version="1.3.1")


class ImagineRequest(BaseModel):
    prompt: str
    img_size: int = 512
    guidance_scale: float = 2.5
    num_inference_steps: int = 24  # menor por defecto: FlowMatch suele tolerarlo
    seed: Optional[int] = None


@app.get("/healthz")
async def healthz():
    return JSONResponse({"status": "ok"})


# =========================== Ingress ===========================
@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, diffusion_model_handle: DeploymentHandle) -> None:
        self.handle = diffusion_model_handle

    @app.get("/debug/token")
    async def debug_token(self):
        present, cuda_visible = await self.handle.get_env_info.remote()
        return JSONResponse(
            {"token_present_in_worker": bool(present), "CUDA_VISIBLE_DEVICES": cuda_visible}
        )

    @app.post(
        "/imagine",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def generate(self, body: ImagineRequest):
        if not body.prompt:
            raise HTTPException(status_code=400, detail="prompt cannot be empty")

        # Sanitizar pasos para evitar sets muy bajos/altos por error
        steps = min(max(int(body.num_inference_steps), 2), 64)

        # Ray Serve agrupa estos kwargs en listas para el método batcheado.
        image = await self.handle.generate.remote(
            prompts=body.prompt,
            img_sizes=int(body.img_size),
            guidance_scales=float(body.guidance_scale),
            num_inference_steps=steps,
            seeds=body.seed,
        )

        buf = BytesIO()
        image.save(buf, "PNG")
        return Response(content=buf.getvalue(), media_type="image/png")


# =========================== Worker ===========================
@serve.deployment(
    autoscaling_config={
        "min_replicas": REPLICAS,
        "max_replicas": REPLICAS,
        "target_ongoing_requests": 1,
    },
    # Evita el warning: max_ongoing_requests >= max_batch_size * max_concurrent_batches (1 por defecto)
    max_ongoing_requests=max(16, BATCH_MAX_SIZE),
    ray_actor_options={
        "num_gpus": ACTOR_GPUS,
        "runtime_env": {
            "env_vars": {
                # Pasamos el token y config de cache al actor
                "HUGGINGFACE_HUB_TOKEN": HF_TOKEN_ENV,
                "HF_HOME": HF_CACHE,
                # Descargas más rápidas (si tenés hf_transfer)
                "HF_HUB_ENABLE_HF_TRANSFER": os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "1"),
                # NCCL para sharding (si usás 2+ GPUs por actor con NVLink/NVLS)
                "NCCL_P2P_DISABLE": os.getenv("NCCL_P2P_DISABLE", "0"),
                "NCCL_NVLS_ENABLE": os.getenv("NCCL_NVLS_ENABLE", "1"),
                "NCCL_P2P_LEVEL": os.getenv("NCCL_P2P_LEVEL", "NVL"),
                "PYTORCH_CUDA_ALLOC_CONF": os.getenv("PYTORCH_CUDA_ALLOC_CONF", DEFAULT_ALLOC_CONF),
            }
        },
    },
)
class FluxKontextEdit:
    def __init__(
        self,
        hf_token: Optional[str] = None,
        use_device_map: bool = USE_DEVICE_MAP,
        use_flash: bool = USE_FLASH,
        use_compile: bool = USE_COMPILE,
    ):
        # IMPORTS locales para evitar capturas no serializables
        from diffusers import FluxPipeline
        from diffusers.utils import load_image
        from huggingface_hub import login
        import torch as _torch  # noqa: F401

        # Token robusto: argumento > env del actor
        token = (hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
        if not token or len(token) < 10:
            raise RuntimeError(
                "HUGGINGFACE_HUB_TOKEN no llegó al actor o es inválido. "
                "Exportalo antes de `serve run` o pasalo en el bind: FluxKontextEdit.bind(hf_token=...)."
            )

        # Login HF (si falla, igual seguimos porque pasamos token a from_pretrained)
        try:
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass

        self._load_image = load_image
        self.autocast_dtype = None  # se setea más abajo
        model_id = "black-forest-labs/FLUX.1-dev"

        # ===== H100-friendly toggles globales =====
        _torch.backends.cuda.matmul.allow_tf32 = True
        _torch.backends.cudnn.allow_tf32 = True
        # SDPA Flash (usa getattr para compatibilidad)
        try:
            sdp_ctx = getattr(_torch.nn.attention, "sdpa_kernel", None)
            if sdp_ctx and use_flash:
                # PyTorch >=2.5
                sdp_ctx(enable_flash=True, enable_math=False, enable_mem_efficient=False)
            elif getattr(_torch.backends.cuda, "sdp_kernel", None) and use_flash:
                # Compat anterior
                _torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        except Exception:
            pass

        # dtype preferido: bf16 en H100
        dtype = _torch.bfloat16
        try:
            _torch.zeros(1, device="cuda", dtype=dtype)
        except Exception:
            dtype = _torch.float16
        self.autocast_dtype = dtype

        # ===== Carga del pipeline (con sharding opcional) =====
        try:
            if use_device_map and ACTOR_GPUS >= 2:
                self.pipe = FluxPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    token=token,
                    cache_dir=HF_CACHE,
                    device_map="balanced",
                )
            else:
                self.pipe = FluxPipeline.from_pretrained(
                    model_id, torch_dtype=dtype, token=token, cache_dir=HF_CACHE
                ).to("cuda")
        except TypeError:
            # Compat versiones viejas
            if use_device_map and ACTOR_GPUS >= 2:
                self.pipe = FluxPipeline.from_pretrained(
                    model_id, torch_dtype=dtype, use_auth_token=token, cache_dir=HF_CACHE
                )
            else:
                self.pipe = FluxPipeline.from_pretrained(
                    model_id, torch_dtype=dtype, use_auth_token=token, cache_dir=HF_CACHE
                ).to("cuda")

        # ===== Afinado de módulos a bf16 + channels_last =====
        for attr in ("text_encoder", "text_encoder_2", "vae", "unet", "transformer"):
            mod = getattr(self.pipe, attr, None)
            if mod is not None:
                try:
                    mod.to(device="cuda", dtype=self.autocast_dtype)
                except Exception:
                    pass

        try:
            if hasattr(self.pipe, "vae"):
                self.pipe.vae.to(memory_format=_torch.channels_last)
            if hasattr(self.pipe, "unet"):
                self.pipe.unet.to(memory_format=_torch.channels_last)
        except Exception:
            pass

        # ===== Optimizaciones adicionales =====
        try:
            self.pipe.enable_vae_slicing()
        except Exception:
            pass
        try:
            if not use_flash and hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        _torch.backends.cudnn.benchmark = True

        # compile (TorchInductor) en el bloque principal
        if use_compile:
            try:
                if hasattr(self.pipe, "transformer"):
                    self.pipe.transformer = _torch.compile(
                        self.pipe.transformer, mode="reduce-overhead", fullgraph=False
                    )
                elif hasattr(self.pipe, "unet"):
                    self.pipe.unet = _torch.compile(
                        self.pipe.unet, mode="reduce-overhead", fullgraph=False
                    )
            except Exception:
                pass

        # ===== Warm-up para amortizar compilación y caches =====
        try:
            with _torch.inference_mode(), _torch.amp.autocast("cuda", dtype=self.autocast_dtype):
                _ = self.pipe(
                    prompt="test",
                    guidance_scale=2.5,
                    num_inference_steps=2,
                    height=256,
                    width=256,
                ).images[0]
        except Exception:
            pass

    async def get_env_info(self):
        return bool((os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()), os.getenv(
            "CUDA_VISIBLE_DEVICES", ""
        )

    @staticmethod
    def _resize_square(img: Image.Image, size: int) -> Image.Image:
        size = max(64, int(size))
        resample = getattr(Image, "Resampling", Image).LANCZOS
        return img.resize((size, size), resample)

    # ============= Micro-batching con fallback =============
    @serve.batch(max_batch_size=BATCH_MAX_SIZE, batch_wait_timeout_s=BATCH_WAIT_MS / 1000.0)
    async def generate(
        self,
        prompts: List[str],
        img_sizes: List[int],
        guidance_scales: List[float],
        num_inference_steps: List[int],
        seeds: List[Optional[int]],
    ):
        import torch as _torch

        # Generators por item (semillas)
        generators: Optional[List[Any]] = []
        any_seed = False
        for s in seeds:
            g = _torch.Generator(device="cuda")
            if s is not None:
                g.manual_seed(int(s))
                any_seed = True
            generators.append(g)

        # Si ninguna seed fue dada, no pases el parámetro `generator`
        if not any_seed:
            generators = None

        steps = [min(max(int(s or 24), 2), 64) for s in num_inference_steps]
        gscales = [float(g) for g in guidance_scales]

        # Nota: FluxPipeline es para text-to-image, no image-to-image
        # Para img2img necesitarías FluxImg2ImgPipeline, pero eso requiere cambios mayores
        # Por ahora, usamos solo text-to-image
        try:
            with _torch.inference_mode(), _torch.amp.autocast("cuda", dtype=self.autocast_dtype):
                out = self.pipe(
                    prompt=prompts,
                    guidance_scale=gscales,
                    num_inference_steps=steps,
                    generator=generators,
                    height=img_sizes,
                    width=img_sizes,
                )
            return out.images  # Ray Serve devolverá el elemento correspondiente a esta solicitud
        except Exception:
            results: List[Image.Image] = []
            for i in range(len(prompts)):
                with _torch.inference_mode(), _torch.amp.autocast("cuda", dtype=self.autocast_dtype):
                    out = self.pipe(
                        prompt=prompts[i],
                        guidance_scale=gscales[i],
                        num_inference_steps=steps[i],
                        generator=None if generators is None else generators[i],
                        height=img_sizes[i],
                        width=img_sizes[i],
                    )
                results.append(out.images[0])
            return results


# =========================== Entrypoint ===========================
entrypoint = APIIngress.bind(
    FluxKontextEdit.bind(
        hf_token=HF_TOKEN_ENV,
        use_device_map=USE_DEVICE_MAP,
        use_flash=USE_FLASH,
        use_compile=USE_COMPILE,
    )
)