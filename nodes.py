import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from server import PromptServer

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - dependency requirement
    raise ImportError(
        "The 'google-generativeai' package is required. Please install it by running 'pip install google-generativeai'."
    )


def tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
    """Convert a ComfyUI IMAGE tensor (N, H, W, C) in [0, 1] to a list of PIL Images."""
    if not isinstance(tensor, torch.Tensor):
        return []

    images: List[Image.Image] = []
    for img_tensor in tensor:
        clipped = torch.clamp(img_tensor, 0.0, 1.0)
        array = (clipped.cpu().numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(array))
    return images


_EXPAND_SYSTEM_INSTRUCTION = """You are an expert prompt engineer for a state-of-the-art image editing model. Your task is to take a user's simple instruction and a set of input images, and expand the instruction into a detailed, descriptive prompt that will guide the image model to produce a high-quality edited result.

Focus on describing the necessary changes based on the user's instruction while considering the content of the original images. Your prompt must include:
(!) Always describe small, specific, targeted edits that will move you to the desired result.

- Visual Style: Match the existing style (e.g., photorealistic, oil painting).
- Composition & Framing: Describe changes in relation to the existing composition.
- Camera: Describe camera parameters and image style.
- Lighting: Describe how lighting should be altered or added, matching the existing light source.
- Details & Texture: Mention specific details from the original image that should be changed.
- Action: Clearly describe the edit to be performed on the image.
- Resulting image: Describe what the result will look like.

The output must be ONLY the new, detailed prompt. Do not add any conversational text, greetings, or explanations."""

_REVIEW_SYSTEM_INSTRUCTION = """You are a self-critical, multimodal AI assistant. Your primary function is to edit images based on user instructions, but a crucial part of your process is to analyze your own work to ensure quality.

You will be shown a conversation history: the original images, the user's instruction, and your previous attempts. Your final task is to analyze your LATEST generated image.

Compare the latest generated image to the original user instruction, considering the entire conversation history for context.

If you have perfectly followed the instruction, respond with only the word 'SUCCESS'.

If you have failed, you must change your strategy. DO NOT repeat the same prompt. Your output must be an updated prompt that points out the previous failure and suggests a corrective action. Do not add conversational text or formatting—your output must be a PROMPT ONLY."""


def _pil_to_inline_image(pil_image: Image.Image) -> Dict[str, Any]:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"inline_data": {"mime_type": "image/png", "data": b64_data}}


def _safe_json_loads(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except json.JSONDecodeError:
        return []
    return []


def _extract_text(response: Any) -> str:
    if not response:
        return ""
    try:
        return response.text
    except AttributeError:
        return ""


def _call_gemini(
    api_key: str,
    model_name: str,
    contents: List[Dict[str, Any]],
    system_instruction: str,
) -> str:
    if api_key:
        genai.configure(api_key=api_key)
    else:
        genai.configure()

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction,
    )

    generation_config = genai.types.GenerationConfig(temperature=0.7)
    response = model.generate_content(contents, generation_config=generation_config)
    return _extract_text(response)


def _build_expand_contents(instruction: str, image: Image.Image) -> List[Dict[str, Any]]:
    image_part = _pil_to_inline_image(image)
    return [{"role": "user", "parts": [image_part, {"text": instruction}]}]


def _build_review_contents(
    instruction: str,
    initial_image: Image.Image,
    latest_image: Image.Image,
    history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    contents: List[Dict[str, Any]] = []
    initial_part = _pil_to_inline_image(initial_image)
    contents.append(
        {
            "role": "user",
            "parts": [initial_part, {"text": f'My instruction is: "{instruction}"'}],
        }
    )

    for step in history:
        image_b64 = step.get("image")
        if isinstance(image_b64, str) and image_b64:
            contents.append(
                {
                    "role": "model",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_b64,
                            }
                        }
                    ],
                }
            )
        analysis = step.get("analysisText")
        if isinstance(analysis, str) and analysis:
            contents.append({"role": "model", "parts": [{"text": analysis}]})

    contents.append({"role": "model", "parts": [_pil_to_inline_image(latest_image)]})
    final_text = (
        "Look at the image and analyze if you failed to follow the instruction. "
        "If you succeeded respond with SUCCESS. If you failed, respond with a corrected prompt."
    )
    contents.append({"role": "user", "parts": [{"text": final_text}]})
    return contents


class DirectorGemini:
    CATEGORY = "Director/Gemini"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "instruction": ("STRING", {"default": "", "multiline": True}),
                "initial_image": ("IMAGE",),
                "model": ("STRING", {"default": "gemini-1.5-pro"}),
                "api_key": ("STRING", {"default": ""}),
                "history_json": ("STRING", {"default": "[]", "multiline": True}),
                "link_id": ("STRING", {"default": ""}),
            },
            "optional": {
                "latest_image": ("IMAGE",),
            },
        }

    def _emit_status(self, link_id: str, done: bool, prompt: str) -> None:
        payload = {"link_id": link_id, "done": done, "prompt": prompt}
        PromptServer.instance.send_sync("director-status", payload)

    def execute(
        self,
        instruction: str,
        initial_image: torch.Tensor,
        model: str,
        api_key: str,
        history_json: str,
        link_id: str,
        latest_image: Optional[torch.Tensor] = None,
    ) -> Tuple[str]:
        prompt = ""

        try:
            initial_images = tensor_to_pil(initial_image)
            if not initial_images:
                raise ValueError("Initial image tensor could not be converted to an image.")
            initial_pil = initial_images[0]

            history = _safe_json_loads(history_json)
            has_latest = latest_image is not None
            contents: List[Dict[str, Any]]
            system_instruction: str

            if has_latest:
                latest_images = tensor_to_pil(latest_image) if isinstance(latest_image, torch.Tensor) else []
                if not latest_images:
                    raise ValueError("Latest image tensor could not be converted to an image.")
                contents = _build_review_contents(instruction, initial_pil, latest_images[0], history)
                system_instruction = _REVIEW_SYSTEM_INSTRUCTION
            else:
                contents = _build_expand_contents(instruction, initial_pil)
                system_instruction = _EXPAND_SYSTEM_INSTRUCTION

            response_text = _call_gemini(api_key, model, contents, system_instruction).strip()

            if has_latest:
                if response_text.upper() == "SUCCESS":
                    prompt = "SUCCESS"
                    self._emit_status(link_id, True, "")
                else:
                    prompt = response_text
                    self._emit_status(link_id, False, prompt)
            else:
                prompt = response_text or instruction
                self._emit_status(link_id, False, prompt)

        except Exception as exc:  # pragma: no cover - defensive path
            prompt = f"<director_error:{exc}>"
            self._emit_status(link_id, True, prompt)

        return (prompt,)


class ImageRouter:
    CATEGORY = "Director/IO"
    RETURN_TYPES: Tuple = ()
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "link_id": ("STRING", {"default": ""}),
                "persist_dir": ("STRING", {"default": "./director_actor"}),
                "iter_tag": ("INT", {"default": 0, "min": 0}),
            }
        }

    def execute(
        self,
        image: torch.Tensor,
        link_id: str,
        persist_dir: str,
        iter_tag: int,
    ) -> Tuple[()]:
        images = tensor_to_pil(image)
        if not images:
            raise ValueError("ImageRouter received an invalid image tensor.")

        directory = Path(persist_dir).expanduser().resolve() / link_id
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"iter_{int(iter_tag):04d}.png"
        path = directory / filename
        images[0].save(path)

        PromptServer.instance.send_sync(
            "img-ready",
            {"link_id": link_id, "path": str(path), "iter": int(iter_tag)},
        )
        return tuple()


NODE_CLASS_MAPPINGS = {
    "DirectorGemini": DirectorGemini,
    "ImageRouter": ImageRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DirectorGemini": "Director (Gemini, single prompt)",
    "ImageRouter": "Image Router (persist + notify)",
}
