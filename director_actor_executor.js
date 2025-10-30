import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    getDirectorActorState,
    getNodesInsideGroup,
    guessOutputNodes,
    sleep,
} from "./director_actor_queue_utils.js";

function makeUuid() {
    if (globalThis.crypto?.randomUUID) {
        return globalThis.crypto.randomUUID();
    }
    return `dae-${Math.random().toString(16).slice(2)}`;
}

app.registerExtension({
    name: "DirectorActorExecutor",
    registerCustomNodes() {
        class DirectorActorExecutorNode {
            constructor() {
                this.serialize_widgets = true;
                this.size = [320, 220];

                const defaults = {
                    directorGroupName: "DirectorGroup",
                    actorGroupName: "ActorGroup",
                    maxLoops: 3,
                    delayMsBetweenPhases: 0,
                    linkId: makeUuid(),
                    isExecuting: false,
                    isCancelling: false,
                    iter: 0,
                    statusText: "Idle",
                };

                this.properties = { ...defaults };

                this.addWidget("text", "Director Group", this.properties.directorGroupName, (value) => {
                    this.properties.directorGroupName = value || "";
                    return this.properties.directorGroupName;
                });
                this.addWidget("text", "Actor Group", this.properties.actorGroupName, (value) => {
                    this.properties.actorGroupName = value || "";
                    return this.properties.actorGroupName;
                });
                this.addWidget("number", "Max Loops", this.properties.maxLoops, (value) => {
                    const numeric = Number(value) || 1;
                    this.properties.maxLoops = Math.max(1, Math.floor(numeric));
                    return this.properties.maxLoops;
                }, { min: 1, step: 1 });
                this.addWidget("number", "Delay (ms)", this.properties.delayMsBetweenPhases, (value) => {
                    const numeric = Number(value) || 0;
                    this.properties.delayMsBetweenPhases = Math.max(0, Math.floor(numeric));
                    return this.properties.delayMsBetweenPhases;
                }, { min: 0, step: 50 });
                this.addWidget("text", "Link ID", this.properties.linkId, (value) => {
                    this.properties.linkId = value || "";
                    return this.properties.linkId;
                });
                this.addWidget("button", "Run", "Run", () => this.startExecution());
                this.addWidget("button", "Cancel", "Cancel", () => this.cancelExecution());
            }

            onExecute() {}

            onDrawForeground(ctx) {
                if (!this.properties?.statusText) {
                    return;
                }
                ctx.save();
                ctx.font = "14px sans-serif";
                ctx.textAlign = "center";
                ctx.fillStyle = this.properties.isCancelling
                    ? "#ff6347"
                    : this.properties.isExecuting
                        ? "#1e90ff"
                        : "#8f8f8f";
                ctx.fillText(this.properties.statusText, this.size[0] / 2, this.size[1] - 18);
                ctx.restore();
            }

            updateStatus(text) {
                this.properties.statusText = text;
                this.setDirtyCanvas(true, true);
            }

            async startExecution() {
                if (this.properties.isExecuting) {
                    console.warn("[DirectorActor] Execution already running");
                    return;
                }
                if (!this.properties.directorGroupName || !this.properties.actorGroupName) {
                    this.updateStatus("Set group names");
                    return;
                }

                this.properties.isExecuting = true;
                this.properties.isCancelling = false;
                this.properties.iter = 0;
                this._lastDirector = null;
                this.updateStatus("Starting...");

                this._cleanupListeners();
                this._subscribeListeners();

                console.debug("[DirectorActor] Starting loop", {
                    directorGroup: this.properties.directorGroupName,
                    actorGroup: this.properties.actorGroupName,
                    linkId: this.properties.linkId,
                });

                try {
                    await this._runLoop();
                    if (this.properties.isCancelling) {
                        this.updateStatus("Cancelled");
                    } else {
                        this.updateStatus("Finished");
                    }
                } catch (error) {
                    console.error("[DirectorActor] Loop failed", error);
                    this.updateStatus(`Error: ${error.message ?? error}`);
                } finally {
                    this.properties.isExecuting = false;
                    this.properties.isCancelling = false;
                    this._cleanupListeners();
                    this.setDirtyCanvas(true, true);
                }
            }

            async cancelExecution() {
                if (!this.properties.isExecuting) {
                    return;
                }
                if (this.properties.isCancelling) {
                    return;
                }
                this.properties.isCancelling = true;
                this.updateStatus("Cancelling...");
                try {
                    await api.fetchApi("/interrupt", { method: "POST" });
                } catch (error) {
                    console.warn("[DirectorActor] Cancel request failed", error);
                }
            }

            async _runLoop() {
                const maxLoops = Math.max(1, Number(this.properties.maxLoops) || 1);
                const delayMs = Math.max(0, Number(this.properties.delayMsBetweenPhases) || 0);

                for (let iteration = 0; iteration < maxLoops; iteration += 1) {
                    if (this.properties.isCancelling) {
                        console.debug("[DirectorActor] Loop cancelled before director phase");
                        break;
                    }

                    this.properties.iter = iteration;
                    this._lastDirector = null;
                    this.updateStatus(`Director phase #${iteration + 1}`);

                    const directorOk = await this.queueGroupAndWait(this.properties.directorGroupName);
                    if (!directorOk) {
                        console.debug("[DirectorActor] Director phase exited early");
                        break;
                    }

                    const directorResult = this._lastDirector;
                    console.debug("[DirectorActor] Director result", directorResult);
                    if (this.properties.isCancelling) {
                        break;
                    }
                    if (directorResult?.done || directorResult?.prompt === "SUCCESS") {
                        console.debug("[DirectorActor] Director signalled completion");
                        break;
                    }

                    this.updateIterationWidget(iteration);
                    this.updateStatus(`Actor phase #${iteration + 1}`);

                    const actorOk = await this.queueGroupAndWait(this.properties.actorGroupName);
                    if (!actorOk) {
                        console.debug("[DirectorActor] Actor phase exited early");
                        break;
                    }

                    if (this.properties.isCancelling) {
                        break;
                    }

                    if (delayMs > 0) {
                        await sleep(delayMs);
                    }
                }
            }

            async queueGroupAndWait(groupName) {
                const nodes = getNodesInsideGroup(groupName);
                if (!nodes.length) {
                    console.warn(`[DirectorActor] No nodes found in group ${groupName}`);
                    return false;
                }
                const outputs = guessOutputNodes(nodes);
                if (!outputs.length) {
                    console.warn(`[DirectorActor] No output nodes detected in group ${groupName}`);
                    return false;
                }

                const state = getDirectorActorState();
                const targetIds = outputs.map((node) => node.id);
                state.targetIds = new Set(targetIds);
                state.active = true;
                state.owner = this.id;

                console.debug("[DirectorActor] Queuing group", groupName, targetIds);

                try {
                    if (window.rgthree?.queueOutputNodes) {
                        await window.rgthree.queueOutputNodes(targetIds);
                    } else if (app.queuePrompt) {
                        await app.queuePrompt();
                    }
                } finally {
                    state.active = false;
                    state.targetIds.clear();
                    state.owner = null;
                }

                const waited = await this.waitForQueueIdle();
                return waited;
            }

            async waitForQueueIdle() {
                while (true) {
                    if (!this.properties.isExecuting || this.properties.isCancelling) {
                        return false;
                    }
                    try {
                        const response = await api.fetchApi("/queue");
                        const data = await response.json();
                        const running = data.queue_running?.length ?? 0;
                        const pending = data.queue_pending?.length ?? 0;
                        if (running === 0 && pending === 0) {
                            return true;
                        }
                    } catch (error) {
                        console.warn("[DirectorActor] Queue poll failed", error);
                    }
                    await sleep(200);
                }
            }

            updateIterationWidget(iterationValue) {
                const nodes = getNodesInsideGroup(this.properties.actorGroupName);
                for (const node of nodes) {
                    if (!Array.isArray(node.widgets)) {
                        continue;
                    }
                    const widget = node.widgets.find((w) => w.name === "iter_tag");
                    if (widget) {
                        widget.value = iterationValue;
                        widget.callback?.(iterationValue);
                        node.setDirtyCanvas?.(true, true);
                    }
                }
            }

            _subscribeListeners() {
                if (!this._unsubscribers) {
                    this._unsubscribers = [];
                }
                const linkId = this.properties.linkId;
                const directorHandler = (event) => {
                    const detail = event?.detail;
                    if (!detail || detail.link_id !== linkId) {
                        return;
                    }
                    this._lastDirector = {
                        done: Boolean(detail.done),
                        prompt: detail.prompt ?? "",
                    };
                    this.updateStatus(detail.done ? "Director done" : "Director prompt ready");
                };
                api.addEventListener("director-status", directorHandler);
                this._unsubscribers.push(() => api.removeEventListener("director-status", directorHandler));

                const imgHandler = (event) => {
                    const detail = event?.detail;
                    if (!detail || detail.link_id !== linkId) {
                        return;
                    }
                    console.debug("[DirectorActor] Image ready", detail);
                };
                api.addEventListener("img-ready", imgHandler);
                this._unsubscribers.push(() => api.removeEventListener("img-ready", imgHandler));

                const interruptHandler = () => {
                    if (this.properties.isExecuting && !this.properties.isCancelling) {
                        this.properties.isCancelling = true;
                        this.updateStatus("Cancelling...");
                    }
                };
                api.addEventListener("execution_interrupt", interruptHandler);
                this._unsubscribers.push(() => api.removeEventListener("execution_interrupt", interruptHandler));
            }

            _cleanupListeners() {
                if (!this._unsubscribers) {
                    return;
                }
                for (const unsubscribe of this._unsubscribers) {
                    try {
                        unsubscribe();
                    } catch (error) {
                        console.warn("[DirectorActor] Listener cleanup failed", error);
                    }
                }
                this._unsubscribers = [];
            }
        }

        DirectorActorExecutorNode.title = "DirectorActorExecutor";
        DirectorActorExecutorNode.type = "DirectorActorExecutor";
        DirectorActorExecutorNode.prototype.getTitle = function () {
            return "DirectorActorExecutor";
        };

        LiteGraph.registerNodeType("Director/DirectorActorExecutor", DirectorActorExecutorNode);
    },
});
