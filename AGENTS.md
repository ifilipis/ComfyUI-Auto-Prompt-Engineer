You are creating a custom ComfyUI node. You must update this document with issues, goals, and tasks

## Issues
- Missing Director/Actor orchestration to coordinate Gemini prompt generation with actor groups and image routing events.

## Goals
- Deliver a single-output Gemini director node and an image router node that emit frontend events for iteration control.
- Provide frontend automation scripts to slice prompts per group, manage queue execution loops, and mirror LG executor behavior.

## Tasks
- [x] Implement `DirectorGemini` and `ImageRouter` nodes with required event signaling and persistence.
- [x] Add frontend executor, prompt filter, and queue utility scripts at the repository root for installation.
- [x] Update `install.py` to deploy the JavaScript files into `web/extensions/DirectorActor/` during installation.
