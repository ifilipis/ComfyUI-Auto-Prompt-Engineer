import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import {
    directorActorState,
    getGroupByName,
    getNodesInsideGroup,
    resolveOutputNodes,
} from "./director_actor_queue_utils.js";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function generateLinkId() {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
        return crypto.randomUUID();
    }
    const random = Math.floor(Math.random() * 1e9);
    return `director-${Date.now()}-${random}`;
}

function ensureInterruptPatch() {
    if (api.fetchApi.__DirectorActorInterruptPatched) {
        return;
    }

    const originalFetch = api.fetchApi;

    const wrapped = async function (url, options = {}) {
        const response = await originalFetch.call(this, url, options);
        if (
            url === "/interrupt" &&
            (options?.method ?? "GET").toUpperCase() === "POST"
        ) {
            api.dispatchEvent(
                new CustomEvent("execution_interrupt", {
                    detail: { timestamp: Date.now() },
                })
            );
        }
        return response;
    };

    wrapped.__DirectorActorPromptFilter = originalFetch.__DirectorActorPromptFilter;
    wrapped.__DirectorActorInterruptPatched = true;

    api.fetchApi = wrapped;
}

class DirectorActorExecutorNode {
    constructor() {
        this.title = "DirectorActorExecutor";
        this.size = [320, 220];
        this.properties = {
            directorGroupName: "DirectorGroup",
            actorGroupName: "ActorGroup",
            maxLoops: 5,
            delayMsBetweenPhases: 0,
            linkId: generateLinkId(),
            isExecuting: false,
            isCancelling: false,
            currentIteration: 0,
        };

        this.statusMessage = "Idle";
        this.__lastDirector = null;

        this.addWidget("text", "Director Group", this.properties.directorGroupName, (value) => {
            this.properties.directorGroupName = value ?? "";
        });
        this.addWidget("text", "Actor Group", this.properties.actorGroupName, (value) => {
            this.properties.actorGroupName = value ?? "";
        });
        this.addWidget("number", "Max Loops", this.properties.maxLoops, (value) => {
            this.properties.maxLoops = Math.max(1, Math.floor(Number(value) || 1));
        }, { min: 1, step: 1 });
        this.addWidget("number", "Delay (ms)", this.properties.delayMsBetweenPhases, (value) => {
            this.properties.delayMsBetweenPhases = Math.max(0, Math.floor(Number(value) || 0));
        }, { min: 0, step: 50 });
        this.addWidget("text", "Link ID", this.properties.linkId, (value) => {
            this.properties.linkId = value ?? "";
        });

        this.addWidget("button", "Run", "", () => this.startExecution());
        this.addWidget("button", "Cancel", "", () => this.cancelExecution());
    }

    startExecution() {
        if (this.properties.isExecuting) {
            return;
        }

        ensureInterruptPatch();

        this.properties.isExecuting = true;
        this.properties.isCancelling = false;
        this.properties.currentIteration = 0;
        this.__lastDirector = null;
        this._setStatus("Running");
        this.updateIterationWidgets(0);
        console.debug("[DirectorActor] Execution started", {
            linkId: this.properties.linkId,
            directorGroup: this.properties.directorGroupName,
            actorGroup: this.properties.actorGroupName,
            maxLoops: this.properties.maxLoops,
        });

        const unsubscribers = [];

        const directorListener = ({ detail }) => {
            if (!detail || detail.link_id !== this.properties.linkId) {
                return;
            }
            this.__lastDirector = {
                done: Boolean(detail.done),
                prompt: detail.prompt ?? "",
            };
            console.debug("[DirectorActor] Director event received", this.__lastDirector);
        };

        api.addEventListener("director-status", directorListener);
        unsubscribers.push(() => api.removeEventListener("director-status", directorListener));

        const interruptListener = () => {
            if (this.properties.isExecuting) {
                this.properties.isCancelling = true;
                this._setStatus("Cancelling...");
                console.debug("[DirectorActor] Interrupt broadcast consumed");
            }
        };

        api.addEventListener("execution_interrupt", interruptListener);
        unsubscribers.push(() => api.removeEventListener("execution_interrupt", interruptListener));

        directorActorState.linkId = this.properties.linkId;

        this.__loopPromise = (async () => {
            try {
                while (this.properties.currentIteration < this.properties.maxLoops) {
                    if (this.properties.isCancelling) {
                        console.debug("[DirectorActor] Loop break due to cancel request");
                        break;
                    }

                    this._setStatus(`Director pass ${this.properties.currentIteration + 1}`);
                    console.debug("[DirectorActor] Queueing director group", {
                        iteration: this.properties.currentIteration,
                        group: this.properties.directorGroupName,
                    });
                    await this.queueGroupAndWait(this.properties.directorGroupName);
                    if (this.properties.isCancelling) {
                        console.debug("[DirectorActor] Cancel detected after director group");
                        break;
                    }

                    const directorResult = this.__lastDirector;
                    if (!directorResult) {
                        console.debug("[DirectorActor] No director result available");
                        this._setStatus("Director did not respond");
                        break;
                    }

                    const promptText = (directorResult.prompt ?? "").trim();
                    if (directorResult.done || promptText === "SUCCESS") {
                        console.debug("[DirectorActor] Director signalled completion", directorResult);
                        this._setStatus("Director finished");
                        break;
                    }

                    this._setStatus(`Actor pass ${this.properties.currentIteration + 1}`);
                    this.updateIterationWidgets(this.properties.currentIteration);
                    console.debug("[DirectorActor] Queueing actor group", {
                        iteration: this.properties.currentIteration,
                        group: this.properties.actorGroupName,
                    });
                    await this.queueGroupAndWait(this.properties.actorGroupName);
                    if (this.properties.isCancelling) {
                        console.debug("[DirectorActor] Cancel detected after actor group");
                        break;
                    }

                    this.properties.currentIteration += 1;
                    console.debug("[DirectorActor] Iteration incremented", {
                        currentIteration: this.properties.currentIteration,
                    });

                    if (this.properties.delayMsBetweenPhases > 0) {
                        console.debug("[DirectorActor] Delay between phases", {
                            delayMs: this.properties.delayMsBetweenPhases,
                        });
                        await sleep(this.properties.delayMsBetweenPhases);
                    }
                }
            } catch (error) {
                console.error("[DirectorActor] execution error", error);
                this._setStatus(`Error: ${error?.message ?? error}`);
            } finally {
                console.debug("[DirectorActor] Execution finished", {
                    cancelled: this.properties.isCancelling,
                    iterations: this.properties.currentIteration,
                });
                directorActorState.active = false;
                directorActorState.targetNodeIds.clear();
                directorActorState.linkId = null;
                this.properties.isExecuting = false;
                this.properties.isCancelling = false;
                this.setDirtyCanvas(true, true);
                unsubscribers.forEach((fn) => fn());
            }
        })();
    }

    async cancelExecution() {
        if (!this.properties.isExecuting || this.properties.isCancelling) {
            return;
        }
        this.properties.isCancelling = true;
        this._setStatus("Cancelling...");
        console.debug("[DirectorActor] Cancel requested explicitly");
        try {
            await api.fetchApi("/interrupt", { method: "POST" });
        } catch (error) {
            console.warn("[DirectorActor] interrupt failed", error);
        }
    }

    async queueGroupAndWait(groupName) {
        const group = getGroupByName(groupName);
        if (!group) {
            throw new Error(`Group '${groupName}' not found`);
        }

        const nodes = getNodesInsideGroup(group);
        const outputs = resolveOutputNodes(nodes);
        console.debug("[DirectorActor] Resolved group nodes", {
            group: groupName,
            nodeCount: nodes.length,
            outputIds: outputs.map((node) => node.id),
        });

        if (!outputs.length) {
            throw new Error(`Group '${groupName}' has no output nodes`);
        }

        directorActorState.targetNodeIds = new Set(outputs.map((node) => String(node.id)));
        directorActorState.active = true;

        try {
            console.debug("[DirectorActor] Triggering queuePrompt");
            await app.queuePrompt();
            await this.waitForQueueIdle();
        } finally {
            directorActorState.active = false;
            directorActorState.targetNodeIds.clear();
        }
    }

    async waitForQueueIdle() {
        while (true) {
            if (this.properties.isCancelling) {
                console.debug("[DirectorActor] Exiting waitForQueueIdle due to cancel");
                return;
            }
            try {
                const response = await api.fetchApi("/queue");
                if (!response.ok) {
                    throw new Error(`Queue status ${response.status}`);
                }
                const data = await response.json();
                const running = data?.queue_running?.length ?? 0;
                const pending = data?.queue_pending?.length ?? 0;
                if (running === 0 && pending === 0) {
                    await sleep(150);
                    console.debug("[DirectorActor] Queue idle detected");
                    return;
                }
            } catch (error) {
                console.warn("[DirectorActor] queue check failed", error);
            }
            await sleep(300);
        }
    }

    updateIterationWidgets(value) {
        const group = getGroupByName(this.properties.actorGroupName);
        if (!group) {
            return;
        }
        const nodes = getNodesInsideGroup(group);
        for (const node of nodes) {
            if (!Array.isArray(node?.widgets)) {
                continue;
            }
            const widget = node.widgets.find((w) => w?.name === "iter_tag");
            if (widget) {
                widget.value = value;
                if (typeof widget.callback === "function") {
                    widget.callback(value);
                }
            }
        }
    }

    _setStatus(text) {
        this.statusMessage = text;
        this.setDirtyCanvas(true, true);
    }

    onDrawForeground(ctx) {
        if (!this.statusMessage) {
            return;
        }
        ctx.save();
        ctx.font = "16px sans-serif";
        ctx.fillStyle = this.properties.isExecuting ? "#1e90ff" : "#888";
        ctx.textAlign = "center";
        ctx.fillText(this.statusMessage, this.size[0] / 2, this.size[1] - 30);
        ctx.restore();
    }
}

LiteGraph.registerNodeType("Director/DirectorActorExecutor", DirectorActorExecutorNode);

app.registerExtension({
    name: "DirectorActorExecutor",
    init() {},
});
