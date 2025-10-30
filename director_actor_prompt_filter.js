import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { buildFilteredPrompt, directorActorState } from "./director_actor_queue_utils.js";

function patchFetchApi() {
    if (api.fetchApi.__DirectorActorPromptFilter) {
        return;
    }

    const originalFetch = api.fetchApi;

    api.fetchApi = async function (url, options = {}) {
        try {
            if (
                directorActorState.active &&
                url === "/prompt" &&
                (options?.method ?? "GET").toUpperCase() === "POST" &&
                typeof options.body === "string"
            ) {
                const payload = JSON.parse(options.body);

                if (!payload.extra_data?.isDirectorActorRequest && directorActorState.targetNodeIds.size > 0) {
                    const filteredPrompt = buildFilteredPrompt(
                        payload.prompt,
                        Array.from(directorActorState.targetNodeIds)
                    );
                    console.debug("[DirectorActor] Filtering prompt payload", {
                        targetCount: directorActorState.targetNodeIds.size,
                        originalNodes: Object.keys(payload.prompt || {}).length,
                        filteredNodes: Object.keys(filteredPrompt || {}).length,
                    });

                    const rewrittenOptions = {
                        ...options,
                        body: JSON.stringify({
                            ...payload,
                            prompt: filteredPrompt,
                            extra_data: {
                                ...payload.extra_data,
                                isDirectorActorRequest: true,
                            },
                        }),
                    };

                    return originalFetch.call(this, url, rewrittenOptions);
                }
            }
        } catch (error) {
            console.warn("[DirectorActor] prompt filter failed", error);
        }

        if (directorActorState.active && url === "/prompt") {
            console.debug("[DirectorActor] Prompt filter passthrough", {
                intercepted: Boolean(directorActorState.active),
                targetCount: directorActorState.targetNodeIds.size,
            });
        }

        return originalFetch.call(this, url, options);
    };

    api.fetchApi.__DirectorActorPromptFilter = true;
}

app.registerExtension({
    name: "DirectorActorPromptFilter",
    setup() {
        patchFetchApi();
    },
    beforeRegisterNodeDef() {
        patchFetchApi();
    },
});
