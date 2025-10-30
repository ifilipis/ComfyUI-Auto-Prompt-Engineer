import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { buildFilteredPrompt, getDirectorActorState } from "./director_actor_queue_utils.js";

app.registerExtension({
    name: "DirectorActorPromptFilter",
    async beforeRegisterNodeDef() {
        if (api.fetchApi.__directorActorFilterPatched) {
            return;
        }

        const originalFetch = api.fetchApi.bind(api);

        api.fetchApi = async (url, options = {}) => {
            const method = (options.method || "GET").toUpperCase();
            const state = getDirectorActorState();

            if (url === "/prompt" && method === "POST" && state.active && state.targetIds.size) {
                try {
                    const bodyText = typeof options.body === "string" ? options.body : await bodyToString(options.body);
                    if (bodyText) {
                        const payload = JSON.parse(bodyText);
                        if (!payload.extra_data?.isDirectorActorRequest) {
                            const filteredPrompt = buildFilteredPrompt(payload.prompt, state.targetIds);
                            let filteredOutput = payload.output;
                            if (filteredOutput && typeof filteredOutput === "object") {
                                const nextOutput = {};
                                for (const id of state.targetIds) {
                                    const key = String(id);
                                    if (key in filteredOutput) {
                                        nextOutput[key] = filteredOutput[key];
                                    }
                                }
                                filteredOutput = nextOutput;
                            }
                            const extra = { ...payload.extra_data, isDirectorActorRequest: true };
                            const newBody = JSON.stringify({
                                ...payload,
                                prompt: filteredPrompt,
                                output: filteredOutput,
                                extra_data: extra,
                            });
                            options = { ...options, body: newBody };
                            console.debug("[DirectorActor] Filtered prompt for targets", Array.from(state.targetIds));
                        }
                    }
                } catch (error) {
                    console.warn("[DirectorActor] Failed to filter prompt", error);
                }
            }

            const response = await originalFetch(url, options);

            if (url === "/interrupt" && method === "POST") {
                queueMicrotask(() => {
                    api.dispatchEvent(new CustomEvent("execution_interrupt"));
                });
            }

            return response;
        };

        api.fetchApi.__directorActorFilterPatched = true;
    },
});

async function bodyToString(body) {
    if (!body) {
        return "";
    }
    if (typeof body === "string") {
        return body;
    }
    if (body instanceof Blob) {
        return await body.text();
    }
    if (body instanceof FormData) {
        const obj = {};
        for (const [key, value] of body.entries()) {
            obj[key] = value;
        }
        return JSON.stringify(obj);
    }
    return "";
}
