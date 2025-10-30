You are creating a custom ComfyUI node. You must update this document with issues, goals, and tasks

## Issues
- Need director/actor orchestration to mirror LG_GroupExecutor behaviour without modifying restricted APIs.

## Goals
- Provide a single-output Gemini director node that emits control events.
- Persist actor images via an Image Router node and notify the front-end.
- Drive group execution loops and prompt slicing purely from front-end extensions.

## Tasks
- Implement `DirectorGemini` and `ImageRouter` classes with required payloads and events.
- Add queue slicing utilities plus `/prompt` interception to scope execution.
- Create an executor node that manages run/cancel loops, queue polling, and debug logging.
