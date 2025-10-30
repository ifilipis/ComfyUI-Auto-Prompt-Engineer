"""ComfyUI nodes for director/actor execution."""

from __future__ import annotations

import base64
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from server import PromptServer

try:
    import google.generativeai as genai
except ImportError as exc:  # pragma: no cover - dependency error surfaced at runtime
    raise ImportError(
        "The 'google-generativeai' package is required. Please install it by running"
        " 'pip install google-generativeai'."
    ) from exc


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


def _tensor_to_pil_list(tensor: torch.Tensor) -> List[Image.Image]:
    if not isinstance(tensor, torch.Tensor):
        return []

    tensor = torch.clamp(tensor, 0.0, 1.0)
    images: List[Image.Image] = []
    for slice_ in tensor:
        array = (slice_.cpu().numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(array))
    return images


def _pil_to_inline_image(image: Image.Image) -> Dict[str, Any]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"inline_data": {"mime_type": "image/png", "data": encoded}}


def _safe_json_list(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(loaded, list):
        return [entry for entry in loaded if isinstance(entry, dict)]
    return []


def _debug_log(message: str, **details: Any) -> None:
    prefix = "[DirectorGemini] "
    if details:
        try:
            serialized = json.dumps(details, default=str)
        except Exception:
            serialized = str(details)
        print(f"{prefix}{message}: {serialized}", flush=True)
    else:
        print(f"{prefix}{message}", flush=True)


def _extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    return getattr(response, "text", "")


def _call_gemini(
    api_key: str,
    model_name: str,
    contents: List[Dict[str, Any]],
    system_instruction: str,
) -> str:
    if api_key:
        genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction,
    )

    generation_config = genai.types.GenerationConfig(temperature=0.7)
    response = model.generate_content(contents, generation_config=generation_config)
    return _extract_response_text(response)


class DirectorGemini:
    """Single-output Gemini director node."""

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

    @classmethod
    def IS_CHANGED(
        cls,
        instruction: str,
        initial_image: torch.Tensor,
        model: str,
        api_key: str,
        history_json: str,
        link_id: str,
        latest_image: Optional[torch.Tensor] = None,
    ) -> float:
        # Force ComfyUI to treat the node as changed every pass so the director reruns.
        return float("nan")

    @staticmethod
    def _build_expand_contents(
        instruction: str,
        initial: Image.Image,
    ) -> Tuple[List[Dict[str, Any]], str]:
        contents = [
            {
                "role": "user",
                "parts": [
                    _pil_to_inline_image(initial),
                    {"text": instruction},
                ],
            }
        ]
        return contents, _EXPAND_SYSTEM_INSTRUCTION

    @staticmethod
    def _add_history_parts(contents: List[Dict[str, Any]], history: Iterable[Dict[str, Any]]) -> None:
        for step in history:
            image_b64 = None
            if isinstance(step, dict):
                image_b64 = step.get("image") or step.get("output") or step.get("initial")
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

            analysis_text = step.get("analysisText") if isinstance(step, dict) else None
            if isinstance(analysis_text, str) and analysis_text.strip():
                contents.append({"role": "model", "parts": [{"text": analysis_text}]})

    @classmethod
    def _build_review_contents(
        cls,
        instruction: str,
        initial: Image.Image,
        latest: Optional[Image.Image],
        history: Iterable[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str]:
        contents: List[Dict[str, Any]] = [
            {
                "role": "user",
                "parts": [
                    _pil_to_inline_image(initial),
                    {"text": f'My instruction is: "{instruction}"'},
                ],
            }
        ]

        cls._add_history_parts(contents, history)

        if latest is not None:
            contents.append({"role": "model", "parts": [_pil_to_inline_image(latest)]})

        contents.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Look at the image and analyze if you failed to follow the instruction. "
                            "If you succeeded, respond with SUCCESS. Otherwise, provide a corrected prompt."
                        )
                    }
                ],
            }
        )

        return contents, _REVIEW_SYSTEM_INSTRUCTION

    @staticmethod
    def _send_event(link_id: str, done: bool, prompt: str) -> None:
        if not link_id:
            return
        _debug_log("Sending director-status", done=done, prompt_preview=prompt[:80])
        try:
            PromptServer.instance.send_sync(
                "director-status",
                {"link_id": link_id, "done": done, "prompt": prompt},
            )
        except Exception:
            # Front-end events should not break execution; swallow errors silently.
            pass

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
        _debug_log(
            "Execute called",
            link_id=link_id,
            has_latest=latest_image is not None,
            instruction_chars=len(instruction or ""),
        )
        prompt_output = ""
        event_prompt = ""
        done = False

        try:
            initial_images = _tensor_to_pil_list(initial_image)
            if not initial_images:
                raise ValueError("Initial image tensor is empty.")
            _debug_log("Initial image processed", count=len(initial_images))

            latest_pil: Optional[Image.Image] = None
            if latest_image is not None:
                latest_list = _tensor_to_pil_list(latest_image)
                if latest_list:
                    latest_pil = latest_list[0]
            _debug_log("Latest image state", available=latest_pil is not None)

            history = _safe_json_list(history_json)
            mode = "expand" if latest_pil is None else "review"
            _debug_log("Resolved mode", mode=mode, history_entries=len(history))

            if mode == "expand":
                contents, system_instruction = self._build_expand_contents(
                    instruction, initial_images[0]
                )
            else:
                contents, system_instruction = self._build_review_contents(
                    instruction, initial_images[0], latest_pil, history
                )
            _debug_log("Built contents", parts=len(contents), system_instruction=system_instruction[:40])

            response_text = _call_gemini(api_key, model, contents, system_instruction).strip()
            _debug_log(
                "Gemini response received",
                empty=not bool(response_text),
                preview=response_text[:80],
            )

            if mode == "expand":
                prompt_output = response_text or instruction
                event_prompt = prompt_output
                done = False
            else:
                if response_text.upper() == "SUCCESS":
                    prompt_output = "SUCCESS"
                    event_prompt = ""
                    done = True
                else:
                    prompt_output = response_text or instruction
                    event_prompt = prompt_output
                    done = False

        except Exception as exc:  # pragma: no cover - runtime safety
            prompt_output = f"<director_error:{exc}>"
            event_prompt = prompt_output
            done = True
            _debug_log("Execute failed", error=str(exc))

        self._send_event(link_id, done, event_prompt)
        _debug_log(
            "Execute completed",
            done=done,
            prompt_preview=prompt_output[:80],
        )
        return (prompt_output,)


class ImageRouter:
    """Persist images and notify the director front-end."""

    CATEGORY = "Director/IO"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
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

    @staticmethod
    def _notify(link_id: str, path: Path, iteration: int) -> None:
        if not link_id:
            return
        payload = {"link_id": link_id, "path": str(path), "iter": iteration}
        try:
            PromptServer.instance.send_sync("img-ready", payload)
        except Exception:
            pass

    def execute(
        self,
        image: torch.Tensor,
        link_id: str,
        persist_dir: str,
        iter_tag: int,
    ) -> Tuple[()]:
        pil_images = _tensor_to_pil_list(image)
        if not pil_images:
            raise ValueError("ImageRouter received an empty image tensor.")

        base_path = Path(os.path.expanduser(persist_dir))
        if link_id:
            base_path = base_path / link_id
        base_path.mkdir(parents=True, exist_ok=True)

        filename = f"iter_{int(iter_tag):04d}.png"
        target_path = base_path / filename
        pil_images[0].save(target_path, format="PNG")

        self._notify(link_id, target_path.resolve(), int(iter_tag))
        return tuple()


NODE_CLASS_MAPPINGS = {
    "DirectorGemini": DirectorGemini,
    "ImageRouter": ImageRouter,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "DirectorGemini": "Director (Gemini, single prompt)",
    "ImageRouter": "Image Router (persist + notify)",
}
