from pathlib import Path
import shutil
from typing import Optional


JS_FILES = [
    "director_actor_executor.js",
    "director_actor_prompt_filter.js",
    "director_actor_queue_utils.js",
]


def _find_web_dir(start: Path) -> Optional[Path]:
    for path in [start, *start.parents]:
        web_dir = path / "web"
        if (web_dir / "scripts" / "app.js").exists():
            return web_dir
    return None


def install():
    """ComfyUI custom node install hook."""
    here = Path(__file__).resolve().parent
    web_dir = _find_web_dir(here)
    if web_dir is None:
        return True

    target_dir = web_dir / "extensions" / "DirectorActor"
    target_dir.mkdir(parents=True, exist_ok=True)

    success = True
    for filename in JS_FILES:
        source = here / filename
        if not source.exists():
            success = False
            continue
        shutil.copy2(source, target_dir / filename)

    return success
