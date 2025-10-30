You are creating a custom ComfyUI node. You must update this document with issues, goals, and tasks

## Issues
- Existing GeminiDirector/PromptSwitch nodes could not orchestrate director/actor loop behaviour.
- Debug logging was missing across director, router, and executor flows.

## Goals
- Provide a single-output DirectorGemini node that communicates completion state via events.
- Persist actor output images and notify the front-end for routing.
- Supply browser code that loops Director and Actor groups with prompt filtering and cancel support.
- Emit lightweight debug output for each execution phase to aid troubleshooting.

## Tasks
- [x] Replace legacy nodes with DirectorGemini and ImageRouter implementations that emit required events.
- [x] Add install hook wiring to expose front-end assets without subdirectories.
- [x] Implement front-end executor, prompt filter, and queue utilities to mirror LG_GroupExecutor behaviour.
- [x] Instrument DirectorGemini, ImageRouter, and front-end scripts with concise debug output.
