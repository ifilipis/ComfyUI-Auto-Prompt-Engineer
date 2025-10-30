import { api } from "../../scripts/api.js";

function getQueueUtils() {
    return window.__DAE_queueUtils;
}

function wrapFetch() {
    if (api.fetchApi.__directorActorWrapped) {
        return;
    }

    const originalFetch = api.fetchApi.bind(api);

    api.fetchApi = async function(url, options = {}) {
        if (url === "/interrupt" && (!options.method || options.method === "POST")) {
            const response = await originalFetch(url, options);
            api.dispatchEvent(new CustomEvent("execution_interrupt"));
            return response;
        }

        if (url === "/prompt" && options?.method === "POST") {
            try {
                const payload = JSON.parse(options.body);
                if (payload?.extra_data?.isDirectorActorRequest) {
                    return originalFetch(url, options);
                }

                const utils = getQueueUtils();
                if (utils?.isActive?.()) {
                    const targets = utils.getTargets();
                    if (targets.size > 0) {
                        const filteredPrompt = utils.buildFilteredPrompt(payload.prompt, targets);
                        const body = JSON.stringify({
                            ...payload,
                            prompt: filteredPrompt,
                            extra_data: {
                                ...payload.extra_data,
                                isDirectorActorRequest: true,
                            },
                        });
                        const nextOptions = { ...options, body };
                        return originalFetch(url, nextOptions);
                    }
                }
            } catch (err) {
                console.warn("[DirectorActor] failed to filter prompt", err);
            }
        }

        return originalFetch(url, options);
    };

    api.fetchApi.__directorActorWrapped = true;
}

wrapFetch();
