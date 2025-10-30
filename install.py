from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

WEB_ASSETS: tuple[str, ...] = (
    "director_actor_executor.js",
    "director_actor_prompt_filter.js",
    "director_actor_queue_utils.js",
)


def _set_web_directory() -> None:
    package_path = Path(__file__).resolve().parent
    candidates: Iterable[str] = (
        package_path.name,
        package_path.name.replace("-", "_"),
    )

    for name in candidates:
        module = sys.modules.get(name)
        if module is None:
            continue
        setattr(module, "WEB_DIRECTORY", ".")
        setattr(module, "WEB_FILES", WEB_ASSETS)
        break


def install() -> bool:
    _set_web_directory()
    try:
        from server import PromptServer  # type: ignore
    except ImportError:
        PromptServer = None  # type: ignore

    if PromptServer is not None:
        root = Path(__file__).resolve().parent
        loader = getattr(PromptServer.instance, "load_js", None) or getattr(
            PromptServer.instance, "load_custom_js", None
        )
        if callable(loader):
            for filename in WEB_ASSETS:
                path = root / filename
                if path.exists():
                    try:
                        loader(str(path))
                    except Exception:
                        pass

    return True
