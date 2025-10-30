import base64
import io
import json
from typing import Any, Dict, List, Tuple

import torch
import numpy as np
from PIL import Image

# --- Gemini API Setup and Helpers ---

# Attempt to import the modern Google Generative AI library.
try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("The 'google-generativeai' package is required. Please install it by running 'pip install google-generativeai'.")

# --- Custom Helper Function for Tensor to PIL Conversion ---

def tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
    """
    Converts a ComfyUI IMAGE tensor (N, H, W, C) in the range [0, 1]
    to a list of PIL Images.
    """
    if not isinstance(tensor, torch.Tensor):
        return []

    images = []
    for img_tensor in tensor:
        # Clamp the values to be safe
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
        # Convert to numpy array, scale to 0-255, and change type to uint8
        image_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        # Create PIL Image from array
        pil_image = Image.fromarray(image_np)
        images.append(pil_image)
    return images


# --- System Instructions for the Gemini Model ---

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

_FORCE_REVIEW_SYSTEM_INSTRUCTION = """You are a self-critical, multimodal AI assistant. You previously generated an image that the user was unsatisfied with, and you must now re-evaluate your work and find the flaw.

Critically re-examine your last generated image against the original user instruction and the entire conversation history. If the user provided specific feedback, you MUST prioritize addressing it.

You MUST find a flaw. Do not determine that you were successful.

Your output must be a new, updated prompt for the image generation model. This prompt should be written as a humble, self-critical analysis in the first person, explaining what you misinterpreted or could have done better. Start by acknowledging the user's dissatisfaction (e.g., "The image is still not quite right because..."). This analysis itself will be used as the next prompt. Do NOT add any other conversational text or formatting. Your output must be a PROMPT ONLY."""


def _pil_to_inline_image(pil_image: Image.Image) -> Dict[str, Any]:
    """Converts a PIL Image to a Gemini API inline_data dictionary."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"inline_data": {"mime_type": "image/png", "data": b64_data}}


def _safe_json_loads(text: str) -> List[Dict[str, Any]]:
    """Safely loads a JSON string into a list of dictionaries, returning an empty list on failure."""
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
    """Extracts text content from a Gemini API response."""
    if not response:
        return ""
    try:
        return response.text
    except AttributeError:
        return ""


def _call_gemini(api_key: str, model_name: str, contents: List[Dict[str, Any]], system_instruction: str) -> str:
    """Calls the Gemini API with the provided content and system instruction."""
    if not api_key:
        raise ValueError("Gemini API key is missing.")
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction
    )
    
    generation_config = genai.types.GenerationConfig(temperature=0.7)
    
    response = model.generate_content(contents, generation_config=generation_config)
    
    return _extract_text(response)


class GeminiDirector:
    """A ComfyUI node that uses the Gemini API to direct image generation."""
    CATEGORY = "Director/Gemini"
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("prompt", "done")
    FUNCTION = "execute"
    OUTPUT_IS_LIST = (False, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["expand", "review", "force_review"],),
                "goal": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "model": ("STRING", {"default": "gemini-1.5-pro-latest"}),
                "api_key": ("STRING", {"default": ""}),
                "history_json": ("STRING", {"default": "[]", "multiline": True}),
            },
            "optional": {
                "feedback": ("STRING", {"default": "", "multiline": True}),
            },
        }

    @staticmethod
    def _build_contents(mode: str, goal: str, current_image: Image.Image, history: List[Dict[str, Any]], feedback: str) -> Tuple[List[Dict[str, Any]], str]:
        """Constructs the 'contents' list for the Gemini API call."""
        current_image_part = _pil_to_inline_image(current_image)

        if mode == "expand":
            contents = [{"role": "user", "parts": [current_image_part, {"text": goal}]}]
            return contents, _EXPAND_SYSTEM_INSTRUCTION

        initial_image_b64 = ""
        if history:
            for step in history:
                img_b64 = step.get("initial") or step.get("input") or step.get("image")
                if isinstance(img_b64, str) and img_b64:
                    initial_image_b64 = img_b64
                    break
        
        if not initial_image_b64:
             initial_image_b64 = current_image_part["inline_data"]["data"]

        initial_image_part = {"inline_data": {"mime_type": "image/png", "data": initial_image_b64}}
        contents = [{"role": "user", "parts": [initial_image_part, {"text": f'My instruction is: "{goal}"'}]}]

        for step in history:
            img_b64 = step.get("image")
            if isinstance(img_b64, str):
                image_part = {"inline_data": {"mime_type": "image/png", "data": img_b64}}
                contents.append({"role": "model", "parts": [image_part]})
            
            analysis = step.get("analysisText")
            if isinstance(analysis, str):
                contents.append({"role": "model", "parts": [{"text": analysis}]})

        contents.append({"role": "model", "parts": [current_image_part]})

        if mode == "review":
            final_text = "Look at this latest image. Analyze if you have perfectly followed the instructions. If so, say SUCCESS. If not, provide an improved prompt."
            contents.append({"role": "user", "parts": [{"text": final_text}]})
            return contents, _REVIEW_SYSTEM_INSTRUCTION
        
        final_feedback = feedback.strip() or "The image still failed to meet the instructions. Re-analyze your work, find the flaw, and provide the detailed, first-person critique as instructed."
        contents.append({"role": "user", "parts": [{"text": final_feedback}]})
        return contents, _FORCE_REVIEW_SYSTEM_INSTRUCTION

    def execute(self, mode: str, goal: str, image: torch.Tensor, model: str, api_key: str, history_json: str, feedback: str = ""):
        prompt = ""
        done = True

        try:
            pil_images = tensor_to_pil(image)
            if not pil_images:
                raise ValueError("Input tensor could not be converted to an image.")
            current_pil_image = pil_images[0]
            
            history = _safe_json_loads(history_json)
            contents, system_instruction = self._build_contents(mode, goal, current_pil_image, history, feedback)
            response_text = _call_gemini(api_key, model, contents, system_instruction).strip()

            if mode == "review" and response_text.upper() == "SUCCESS":
                prompt = ""
                done = True
            else:
                prompt = response_text
                done = not bool(prompt)

        except Exception as e:
            prompt = f"<error> GeminiDirector failed: {e}"
            done = True
            print(prompt)

        return (prompt, done)


class PromptSwitch:
    """A utility node to switch between an initial prompt and subsequent prompts."""
    CATEGORY = "Director/Gemini"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("current_prompt",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "iteration": ("INT", {"default": 0, "min": 0}),
                "first_prompt": ("STRING", {"default": "", "multiline": True}),
                "next_prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def execute(self, iteration: int, first_prompt: str, next_prompt: str):
        current_prompt = first_prompt if iteration == 0 else next_prompt
        return (current_prompt,)


# --- Node Mappings for ComfyUI ---

NODE_CLASS_MAPPINGS = {
    "GeminiDirector": GeminiDirector,
    "PromptSwitch": PromptSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiDirector": "Gemini Director",
    "PromptSwitch": "Prompt Switch",
}
