"""ComfyUI nodes for Qwen3-VL models."""

from __future__ import annotations

import gc
import importlib.util
import json
import platform
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

import folder_paths

_MODEL_CONFIG_PATH = Path(__file__).with_name("qwen3_vl_models.json")
_DEFAULT_PROMPTS = ["Describe this image in detail."]

QWEN3_VL_MODELS: Dict[str, Dict[str, Any]] = {}


class Quantization(str, Enum):
    Q4 = "4-bit (VRAM-friendly)"
    Q8 = "8-bit (Balanced)"
    FP16 = "None (FP16)"

    @classmethod
    def get_values(cls) -> List[str]:
        return [item.value for item in cls]

    @classmethod
    def from_value(cls, value: str) -> "Quantization":
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"Unsupported quantization: {value}")


ATTENTION_MODES = ["auto", "flash_attention_2", "sdpa"]


def _load_model_configs() -> Dict[str, Dict[str, Any]]:
    try:
        with _MODEL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle) or {}
        models = data.get("qwen3_vl_models")
        if isinstance(models, dict) and models:
            return models
    except FileNotFoundError:
        pass
    except json.JSONDecodeError as exc:
        print(f"[Qwen3-VL] Model config parse failed: {exc}")
    return {
        "Qwen3-VL-4B-Instruct": {
            "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
            "default": True,
            "quantized": False,
            "vram_requirement": {"full": 6.0, "8bit": 3.5, "4bit": 2.0},
        }
    }


if not QWEN3_VL_MODELS:
    QWEN3_VL_MODELS = _load_model_configs()


def _require_transformers():
    if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("huggingface_hub") is None:
        raise ImportError(
            "Qwen3-VL nodes require 'transformers' and 'huggingface_hub'. "
            "Install them with 'pip install transformers huggingface_hub'."
        )
    from huggingface_hub import snapshot_download
    from transformers import (
        AutoModelForVision2Seq,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    return AutoModelForVision2Seq, AutoProcessor, AutoTokenizer, BitsAndBytesConfig, snapshot_download


def _get_device_info() -> Dict[str, Any]:
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        return {
            "gpu": {
                "available": True,
                "total_memory": total_mem / 1024**3,
                "free_memory": free_mem / 1024**3,
            },
            "recommended_device": "cuda",
        }
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return {"gpu": {"available": True, "total_memory": 0, "free_memory": 0}, "recommended_device": "mps"}
    return {"gpu": {"available": False, "total_memory": 0, "free_memory": 0}, "recommended_device": "cpu"}


def _normalize_device_choice(device: str) -> str:
    device = (device or "auto").strip()
    if device == "auto":
        return "auto"
    if device.isdigit():
        device = f"cuda:{int(device)}"
    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            return "cpu"
        if ":" in device:
            try:
                device_idx = int(device.split(":", 1)[1])
                if device_idx >= torch.cuda.device_count():
                    return "cuda:0"
            except (ValueError, IndexError):
                return "cuda:0"
        return device
    if device == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _flash_attn_available() -> bool:
    if platform.system() != "Linux" or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return False
    return importlib.util.find_spec("flash_attn") is not None


def _resolve_attention_mode(mode: str) -> str:
    if mode == "sdpa":
        return "sdpa"
    if mode == "flash_attention_2":
        if _flash_attn_available():
            return "flash_attention_2"
        print("[Qwen3-VL] Flash-Attn forced but unavailable, falling back to SDPA")
        return "sdpa"
    if _flash_attn_available():
        return "flash_attention_2"
    return "sdpa"


def _ensure_model(model_name: str) -> str:
    _, _, _, _, snapshot_download = _require_transformers()
    info = QWEN3_VL_MODELS.get(model_name)
    if not info:
        raise ValueError(f"Model '{model_name}' not in config")
    repo_id = info["repo_id"]

    llm_paths = folder_paths.get_folder_paths("LLM") if "LLM" in folder_paths.folder_names_and_paths else []
    if llm_paths:
        models_dir = Path(llm_paths[0]) / "Qwen-VL"
    else:
        models_dir = Path(folder_paths.models_dir) / "LLM" / "Qwen-VL"

    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / repo_id.split("/")[-1]

    if target.exists() and target.is_dir():
        if any(target.glob("*.safetensors")) or any(target.glob("*.bin")):
            return str(target)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", ".git*"],
    )
    return str(target)


def _enforce_memory(model_name: str, quantization: Quantization, device_info: Dict[str, Any]) -> Quantization:
    info = QWEN3_VL_MODELS.get(model_name, {})
    requirements = info.get("vram_requirement", {})
    mapping = {
        Quantization.Q4: requirements.get("4bit", 0),
        Quantization.Q8: requirements.get("8bit", 0),
        Quantization.FP16: requirements.get("full", 0),
    }
    needed = mapping.get(quantization, 0)
    if not needed:
        return quantization
    available = device_info.get("gpu", {}).get("free_memory", 0)
    if needed * 1.2 > available and available > 0:
        if quantization == Quantization.FP16:
            print("[Qwen3-VL] Auto-switch to 8-bit due to VRAM pressure")
            return Quantization.Q8
        if quantization == Quantization.Q8:
            print("[Qwen3-VL] Auto-switch to 4-bit due to VRAM pressure")
            return Quantization.Q4
    return quantization


def _quantization_config(quantization: Quantization):
    _, _, _, BitsAndBytesConfig, _ = _require_transformers()
    if quantization == Quantization.Q4:
        return (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ),
            None,
        )
    if quantization == Quantization.Q8:
        return BitsAndBytesConfig(load_in_8bit=True), None
    return None, torch.float16 if torch.cuda.is_available() else torch.float32


def _tensor_to_pil(tensor: torch.Tensor) -> Optional[Image.Image]:
    if tensor is None:
        return None
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


class Qwen3VLBase:
    def __init__(self) -> None:
        self.device_info = _get_device_info()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature: Optional[Tuple[Any, ...]] = None
        print("[Qwen3-VL] Node ready")

    def clear(self) -> None:
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(
        self,
        model_name: str,
        quant_value: str,
        attention_mode: str,
        use_compile: bool,
        device_choice: str,
        keep_model_loaded: bool,
    ) -> None:
        AutoModelForVision2Seq, AutoProcessor, AutoTokenizer, _, _ = _require_transformers()
        quant = _enforce_memory(model_name, Quantization.from_value(quant_value), self.device_info)
        attn_impl = _resolve_attention_mode(attention_mode)
        device_requested = self.device_info["recommended_device"] if device_choice == "auto" else device_choice
        device = _normalize_device_choice(device_requested)
        signature = (model_name, quant.value, attn_impl, device, use_compile)
        if keep_model_loaded and self.model is not None and self.current_signature == signature:
            return
        self.clear()
        model_path = _ensure_model(model_name)
        quant_config, dtype = _quantization_config(quant)
        load_kwargs: Dict[str, Any] = {
            "device_map": device if device != "auto" else "auto",
            "attn_implementation": attn_impl,
            "use_safetensors": True,
            "trust_remote_code": True,
        }
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        if quant_config:
            load_kwargs["quantization_config"] = quant_config
        print(f"[Qwen3-VL] Loading {model_name} ({quant.value}, attn={attn_impl})")
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).eval()
        self.model.config.use_cache = True
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = True
        if use_compile and device.startswith("cuda") and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[Qwen3-VL] torch.compile enabled")
            except Exception as exc:
                print(f"[Qwen3-VL] torch.compile skipped: {exc}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.current_signature = signature

    @torch.no_grad()
    def generate(
        self,
        prompt_text: str,
        image: Optional[torch.Tensor],
        video: Optional[torch.Tensor],
        frame_count: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        num_beams: int,
        repetition_penalty: float,
    ) -> str:
        conversation = [{"role": "user", "content": []}]
        if image is not None:
            conversation[0]["content"].append({"type": "image", "image": _tensor_to_pil(image)})
        if video is not None:
            frames = []
            for frame in video:
                pil_frame = _tensor_to_pil(frame)
                if pil_frame is not None:
                    frames.append(pil_frame)
            if len(frames) > frame_count:
                idx = np.linspace(0, len(frames) - 1, frame_count, dtype=int)
                frames = [frames[i] for i in idx]
            if frames:
                conversation[0]["content"].append({"type": "video", "video": frames})
        conversation[0]["content"].append({"type": "text", "text": prompt_text})
        chat = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        images = [item["image"] for item in conversation[0]["content"] if item["type"] == "image"]
        video_frames = [
            frame
            for item in conversation[0]["content"]
            if item["type"] == "video"
            for frame in item["video"]
        ]
        videos = [video_frames] if video_frames else None
        processed = self.processor(text=chat, images=images or None, videos=videos, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        model_inputs = {
            key: value.to(model_device) if torch.is_tensor(value) else value
            for key, value in processed.items()
        }
        stop_tokens = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "eot_id") and self.tokenizer.eot_id is not None:
            stop_tokens.append(self.tokenizer.eot_id)
        kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
            "eos_token_id": stop_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if num_beams == 1:
            kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        else:
            kwargs["do_sample"] = False
        outputs = self.model.generate(**model_inputs, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        input_len = model_inputs["input_ids"].shape[-1]
        text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
        return text.strip()

    def run(
        self,
        model_name: str,
        quantization: str,
        prompt: str,
        custom_prompt: str,
        image: Optional[torch.Tensor],
        video: Optional[torch.Tensor],
        frame_count: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        num_beams: int,
        repetition_penalty: float,
        seed: int,
        keep_model_loaded: bool,
        attention_mode: str,
        use_torch_compile: bool,
        device: str,
    ) -> Tuple[str]:
        torch.manual_seed(seed)
        final_prompt = prompt or _DEFAULT_PROMPTS[0]
        if custom_prompt and custom_prompt.strip():
            final_prompt = custom_prompt.strip()
        self.load_model(
            model_name,
            quantization,
            attention_mode,
            use_torch_compile,
            device,
            keep_model_loaded,
        )
        try:
            text = self.generate(
                final_prompt,
                image,
                video,
                frame_count,
                max_tokens,
                temperature,
                top_p,
                num_beams,
                repetition_penalty,
            )
            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()


class Qwen3VL(Qwen3VLBase):
    @classmethod
    def INPUT_TYPES(cls):
        models = list(QWEN3_VL_MODELS.keys())
        default_model = next((name for name, info in QWEN3_VL_MODELS.items() if info.get("default")), None)
        default_model = default_model or (models[0] if models else "Qwen3-VL-4B-Instruct")
        return {
            "required": {
                "model_name": (models, {"default": default_model}),
                "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto"}),
                "prompt": ("STRING", {"default": _DEFAULT_PROMPTS[0], "multiline": True}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "AutoPromptEngineer"

    def process(
        self,
        model_name: str,
        quantization: str,
        attention_mode: str,
        prompt: str,
        custom_prompt: str,
        max_tokens: int,
        keep_model_loaded: bool,
        seed: int,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
    ) -> Tuple[str]:
        return self.run(
            model_name,
            quantization,
            prompt,
            custom_prompt,
            image,
            video,
            16,
            max_tokens,
            0.6,
            0.9,
            1,
            1.2,
            seed,
            keep_model_loaded,
            attention_mode,
            False,
            "auto",
        )


class Qwen3VLAdvanced(Qwen3VLBase):
    @classmethod
    def INPUT_TYPES(cls):
        models = list(QWEN3_VL_MODELS.keys())
        default_model = next((name for name, info in QWEN3_VL_MODELS.items() if info.get("default")), None)
        default_model = default_model or (models[0] if models else "Qwen3-VL-4B-Instruct")

        num_gpus = torch.cuda.device_count()
        gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
        device_options = ["auto", "cpu", "mps"] + gpu_list

        return {
            "required": {
                "model_name": (models, {"default": default_model}),
                "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto"}),
                "use_torch_compile": ("BOOLEAN", {"default": False}),
                "device": (device_options, {"default": "auto"}),
                "prompt": ("STRING", {"default": _DEFAULT_PROMPTS[0], "multiline": True}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.0}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "AutoPromptEngineer"

    def process(
        self,
        model_name: str,
        quantization: str,
        attention_mode: str,
        use_torch_compile: bool,
        device: str,
        prompt: str,
        custom_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        num_beams: int,
        repetition_penalty: float,
        frame_count: int,
        keep_model_loaded: bool,
        seed: int,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
    ) -> Tuple[str]:
        return self.run(
            model_name,
            quantization,
            prompt,
            custom_prompt,
            image,
            video,
            frame_count,
            max_tokens,
            temperature,
            top_p,
            num_beams,
            repetition_penalty,
            seed,
            keep_model_loaded,
            attention_mode,
            use_torch_compile,
            device,
        )


NODE_CLASS_MAPPINGS = {
    "Qwen3VL": Qwen3VL,
    "Qwen3VLAdvanced": Qwen3VLAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VL": "Qwen3-VL",
    "Qwen3VLAdvanced": "Qwen3-VL (Advanced)",
}
