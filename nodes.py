import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image

try:
    import comfy.utils  # type: ignore
except ImportError:  # pragma: no cover - ComfyUI runtime provides this
    comfy = None  # type: ignore
else:
    comfy = comfy.utils  # type: ignore

try:
    from server import PromptServer  # type: ignore
except ImportError:  # pragma: no cover - when ComfyUI server not available
    PromptServer = None  # type: ignore

try:
    import google.generativeai as genai
except ImportError as exc:
    raise ImportError(
        "The 'google-generativeai' package is required. Please install it by running 'pip install google-generativeai'."
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


def _tensor_to_pils(tensor: Optional[torch.Tensor]) -> List[Image.Image]:
    if tensor is None:
        return []
    if comfy is not None:
        return comfy.tensor_to_pil(tensor)
    if not isinstance(tensor, torch.Tensor):
        return []
    images: List[Image.Image] = []
    for item in tensor:
        item = torch.clamp(item, 0.0, 1.0)
        array = (item.cpu().numpy() * 255).astype("uint8")
        images.append(Image.fromarray(array))
    return images


def _pil_to_inline(pil_image: Image.Image) -> Dict[str, Any]:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"inline_data": {"mime_type": "image/png", "data": encoded}}


def _safe_json_list(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _configure_genai(api_key: str) -> None:
    if api_key:
        genai.configure(api_key=api_key)
    else:
        genai.configure()


def _debug(component: str, message: str) -> None:
    print(f"[{component}] {message}")


def _send_event(name: str, payload: Dict[str, Any]) -> None:
    if PromptServer is None:
        return
    try:
        PromptServer.instance.send_sync(name, payload)
    except Exception:
        pass


class DirectorGemini:
    CATEGORY = "Director/Gemini"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "execute"
    OUTPUT_IS_LIST = (False,)

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

    @staticmethod
    def _build_expand_contents(instruction: str, initial: Image.Image) -> Tuple[List[Dict[str, Any]], str]:
        contents = [
            {
                "role": "user",
                "parts": [
                    _pil_to_inline(initial),
                    {"text": instruction},
                ],
            }
        ]
        return contents, _EXPAND_SYSTEM_INSTRUCTION

    @staticmethod
    def _build_review_contents(
        instruction: str,
        initial: Image.Image,
        latest: Image.Image,
        history: Iterable[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str]:
        contents: List[Dict[str, Any]] = [
            {
                "role": "user",
                "parts": [
                    _pil_to_inline(initial),
                    {"text": f'My instruction is: "{instruction}"'},
                ],
            }
        ]

        for step in history:
            image_b64 = step.get("image") or step.get("output")
            if isinstance(image_b64, str) and image_b64:
                contents.append(
                    {
                        "role": "model",
                        "parts": [
                            {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                        ],
                    }
                )
            text = step.get("analysisText") or step.get("prompt")
            if isinstance(text, str) and text.strip():
                contents.append({"role": "model", "parts": [{"text": text}]})

        contents.append({"role": "model", "parts": [_pil_to_inline(latest)]})
        contents.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Look at the image and analyze if you failed to satisfy the instruction. "
                            "If you were perfect, answer with SUCCESS. Otherwise, provide a corrective prompt."
                        )
                    }
                ],
            }
        )
        return contents, _REVIEW_SYSTEM_INSTRUCTION

    @staticmethod
    def _call_gemini(
        api_key: str,
        model_name: str,
        contents: List[Dict[str, Any]],
        system_instruction: str,
    ) -> str:
        _configure_genai(api_key)
        model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)
        response = model.generate_content(contents)
        text = getattr(response, "text", "")
        return text.strip() if isinstance(text, str) else ""

    def execute(
        self,
        instruction: str,
        initial_image: torch.Tensor,
        model: str,
        api_key: str,
        history_json: str,
        link_id: str,
        latest_image: Optional[torch.Tensor] = None,
    ):
        done = False
        prompt_text = ""

        event_prompt = ""

        try:
            _debug("DirectorGemini", f"Starting execution (link_id={link_id or 'N/A'})")
            initial_pils = _tensor_to_pils(initial_image)
            if not initial_pils:
                raise ValueError("Initial image tensor is empty.")
            initial_pil = initial_pils[0]

            latest_pil: Optional[Image.Image] = None
            latest_list = _tensor_to_pils(latest_image) if latest_image is not None else []
            if latest_list:
                latest_pil = latest_list[0]

            history = _safe_json_list(history_json)
            is_review = latest_pil is not None
            _debug("DirectorGemini", f"Mode determined: {'review' if is_review else 'expand'}")

            if is_review:
                contents, system_instruction = self._build_review_contents(
                    instruction,
                    initial_pil,
                    latest_pil,
                    history,
                )
                _debug(
                    "DirectorGemini",
                    f"Review history length: {len(history)}; latest image present={latest_pil is not None}",
                )
            else:
                contents, system_instruction = self._build_expand_contents(
                    instruction,
                    initial_pil,
                )
                _debug("DirectorGemini", "Prepared expand contents with initial image")

            response_text = self._call_gemini(api_key, model, contents, system_instruction)
            _debug("DirectorGemini", f"Gemini response received (chars={len(response_text)})")

            if not is_review:
                prompt_text = response_text or instruction
                done = False
                event_prompt = prompt_text
                _debug("DirectorGemini", "Expand prompt ready")
            else:
                if response_text.strip().upper() == "SUCCESS":
                    done = True
                    prompt_text = "SUCCESS"
                    event_prompt = ""
                    _debug("DirectorGemini", "Review signalled SUCCESS")
                else:
                    prompt_text = response_text or instruction
                    done = False
                    event_prompt = prompt_text
                    _debug("DirectorGemini", "Review produced corrective prompt")

        except Exception as exc:  # pragma: no cover - defensive for runtime issues
            prompt_text = f"<director_error:{exc}>"
            done = True
            event_prompt = prompt_text
            _debug("DirectorGemini", f"error: {exc}")
        finally:
            _debug(
                "DirectorGemini",
                f"Emitting director-status event (done={done}, prompt_length={len(event_prompt)})",
            )
            _send_event(
                "director-status",
                {"link_id": link_id, "done": done, "prompt": event_prompt},
            )

        return (prompt_text,)


class ImageRouter:
    CATEGORY = "Director/IO"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    OUTPUT_IS_LIST = (False,)

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

    def execute(self, image: torch.Tensor, link_id: str, persist_dir: str, iter_tag: int):
        _debug(
            "ImageRouter",
            f"Received image for routing (link_id={link_id or 'N/A'}, iter={int(iter_tag)})",
        )
        pils = _tensor_to_pils(image)
        if not pils:
            raise ValueError("ImageRouter received an empty tensor.")
        pil = pils[0]

        safe_link = link_id or "default"
        target_dir = Path(os.path.expanduser(persist_dir)).resolve() / safe_link
        target_dir.mkdir(parents=True, exist_ok=True)

        filename = f"iter_{int(iter_tag):04d}.png"
        target_path = target_dir / filename
        pil.save(target_path)
        _debug("ImageRouter", f"Saved image to {target_path}")

        _send_event(
            "img-ready",
            {"link_id": link_id, "path": str(target_path), "iter": int(iter_tag)},
        )
        _debug("ImageRouter", "Emitted img-ready event")

        return (image,)


NODE_CLASS_MAPPINGS = {
    "DirectorGemini": DirectorGemini,
    "ImageRouter": ImageRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DirectorGemini": "Director (Gemini, single prompt)",
    "ImageRouter": "Image Router (persist + notify)",
}
