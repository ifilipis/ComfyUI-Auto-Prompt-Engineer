import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function randomId() {
    if (globalThis.crypto?.randomUUID) {
        return crypto.randomUUID();
    }
    return `dae-${Math.random().toString(36).slice(2, 10)}`;
}

function queueUtils() {
    return window.__DAE_queueUtils;
}

function updateWidgets(nodes, name, value) {
    for (const node of nodes) {
        if (!node?.widgets) {
            continue;
        }
        for (const widget of node.widgets) {
            if (widget?.name === name && widget.value !== value) {
                widget.value = value;
                widget.callback?.(value);
                node.setDirtyCanvas(true, true);
            }
        }
    }
}

class DirectorActorExecutorNode extends LiteGraph.LGraphNode {
    constructor() {
        super("Director Actor Executor");
        this.serialize_widgets = true;
        this.size = this.computeSize();
        this.lastDirectorStatus = null;
        this.executionMessage = "";

        this.addProperty("directorGroupName", "", "string");
        this.addProperty("actorGroupName", "", "string");
        this.addProperty("maxLoops", 5, "number");
        this.addProperty("delayMsBetweenPhases", 0, "number");
        this.addProperty("link_id", randomId(), "string");
        this.addProperty("isExecuting", false, "boolean");
        this.addProperty("isCancelling", false, "boolean");
        this.addProperty("iteration", 0, "number");

        this.directorWidget = this.addWidget("text", "Director Group", this.properties.directorGroupName, (value) => {
            this.properties.directorGroupName = value || "";
        });
        this.actorWidget = this.addWidget("text", "Actor Group", this.properties.actorGroupName, (value) => {
            this.properties.actorGroupName = value || "";
        });
        this.maxLoopsWidget = this.addWidget("number", "Max Loops", this.properties.maxLoops, (value) => {
            const parsed = parseInt(value, 10);
            this.properties.maxLoops = Number.isFinite(parsed) && parsed > 0 ? parsed : 1;
        }, { min: 1, step: 1 });
        this.delayWidget = this.addWidget("number", "Delay (ms)", this.properties.delayMsBetweenPhases, (value) => {
            const parsed = parseInt(value, 10);
            this.properties.delayMsBetweenPhases = Number.isFinite(parsed) && parsed >= 0 ? parsed : 0;
        }, { min: 0, step: 100 });
        this.linkWidget = this.addWidget("text", "Link ID", this.properties.link_id, (value) => {
            this.properties.link_id = value || this.properties.link_id || randomId();
            this.linkWidget.value = this.properties.link_id;
        });
        this.addWidget("button", "Run", null, () => this.startExecution());
        this.addWidget("button", "Cancel", null, () => this.cancelExecution());
    }

    computeSize() {
        return [320, 240];
    }

    startExecution() {
        if (this.properties.isExecuting) {
            return;
        }
        const directorGroup = this.properties.directorGroupName?.trim();
        const actorGroup = this.properties.actorGroupName?.trim();
        if (!directorGroup || !actorGroup) {
            console.warn("[DirectorActor] Director or Actor group missing.");
            return;
        }
        this.properties.isExecuting = true;
        this.properties.isCancelling = false;
        this.executionMessage = "Running";
        this.lastDirectorStatus = null;
        this.subscribeEvents();
        this.setDirtyCanvas(true, true);
        this.executeLoop();
    }

    async executeLoop() {
        const utils = queueUtils();
        if (!utils) {
            console.warn("[DirectorActor] queue utils unavailable.");
            this.finishExecution("Missing queue utils");
            return;
        }

        this.syncLinkWidgets();

        const maxLoops = Math.max(1, parseInt(this.properties.maxLoops, 10) || 1);
        let iteration = 0;

        while (iteration < maxLoops && !this.properties.isCancelling) {
            this.lastDirectorStatus = null;
            this.properties.iteration = iteration;
            this.executionMessage = `Running #${iteration + 1}`;
            this.setDirtyCanvas(true, true);

            const directorQueued = await this.queueGroupAndWait(this.properties.directorGroupName);
            if (!directorQueued || this.properties.isCancelling) {
                break;
            }

            const status = this.lastDirectorStatus;
            if (!status) {
                console.warn("[DirectorActor] Director group finished without status event.");
            } else {
                if (status.done || status.prompt === "SUCCESS") {
                    break;
                }
            }

            this.updateIterWidgets(iteration);
            const actorQueued = await this.queueGroupAndWait(this.properties.actorGroupName);
            if (!actorQueued || this.properties.isCancelling) {
                break;
            }

            iteration += 1;
            this.properties.iteration = iteration;
            if (this.properties.delayMsBetweenPhases > 0) {
                await sleep(this.properties.delayMsBetweenPhases);
            }
        }

        const finalMessage = this.properties.isCancelling ? "Cancelled" : "Idle";
        this.finishExecution(finalMessage);
    }

    async queueGroupAndWait(groupName) {
        const utils = queueUtils();
        if (!utils) {
            return false;
        }
        const nodes = utils.outputNodesForGroup(groupName);
        if (!nodes?.length) {
            console.warn(`[DirectorActor] No output nodes in group: ${groupName}`);
            return false;
        }

        utils.activate(nodes.map((node) => node.id));

        try {
            if (window.rgthree?.queueOutputNodes) {
                await window.rgthree.queueOutputNodes(nodes.map((node) => node.id));
            } else {
                for (const node of nodes) {
                    if (typeof node.triggerQueue === "function") {
                        node.triggerQueue();
                    }
                }
            }
        } catch (err) {
            console.error(`[DirectorActor] Failed to queue group ${groupName}`, err);
            utils.deactivate();
            return false;
        }

        const ok = await this.waitForQueueIdle();
        utils.deactivate();
        return ok;
    }

    async waitForQueueIdle() {
        while (true) {
            if (this.properties.isCancelling) {
                return false;
            }
            try {
                const response = await api.fetchApi("/queue");
                if (!response?.ok) {
                    throw new Error(`Queue request failed: ${response?.status}`);
                }
                const data = await response.json();
                const running = data?.queue_running?.length || 0;
                const pending = data?.queue_pending?.length || 0;
                if (running === 0 && pending === 0) {
                    return true;
                }
            } catch (err) {
                console.warn("[DirectorActor] queue poll failed", err);
            }
            await sleep(150);
        }
    }

    updateIterWidgets(iteration) {
        const utils = queueUtils();
        if (!utils) {
            return;
        }
        const nodes = utils.nodesInGroup(this.properties.actorGroupName);
        updateWidgets(nodes, "iter_tag", iteration);
        updateWidgets(nodes, "link_id", this.properties.link_id);
    }

    syncLinkWidgets() {
        const utils = queueUtils();
        if (!utils) {
            return;
        }
        const directorNodes = utils.nodesInGroup(this.properties.directorGroupName);
        const actorNodes = utils.nodesInGroup(this.properties.actorGroupName);
        updateWidgets(directorNodes, "link_id", this.properties.link_id);
        updateWidgets(actorNodes, "link_id", this.properties.link_id);
    }

    async cancelExecution() {
        if (!this.properties.isExecuting || this.properties.isCancelling) {
            return;
        }
        this.properties.isCancelling = true;
        this.executionMessage = "Cancelling";
        this.setDirtyCanvas(true, true);
        try {
            await api.fetchApi("/interrupt", { method: "POST" });
        } catch (err) {
            console.error("[DirectorActor] cancel failed", err);
        }
    }

    subscribeEvents() {
        if (this._directorHandler) {
            return;
        }
        this._directorHandler = ({ detail }) => {
            if (!detail || detail.link_id !== this.properties.link_id) {
                return;
            }
            this.lastDirectorStatus = detail;
        };
        this._interruptHandler = () => {
            if (this.properties.isExecuting && !this.properties.isCancelling) {
                this.properties.isCancelling = true;
            }
        };
        api.addEventListener("director-status", this._directorHandler);
        api.addEventListener("execution_interrupt", this._interruptHandler);
    }

    unsubscribeEvents() {
        if (this._directorHandler) {
            api.removeEventListener("director-status", this._directorHandler);
            this._directorHandler = null;
        }
        if (this._interruptHandler) {
            api.removeEventListener("execution_interrupt", this._interruptHandler);
            this._interruptHandler = null;
        }
    }

    finishExecution(message) {
        this.unsubscribeEvents();
        queueUtils()?.deactivate();
        this.properties.isExecuting = false;
        this.properties.isCancelling = false;
        this.executionMessage = message || "";
        this.setDirtyCanvas(true, true);
    }

    onRemoved() {
        this.unsubscribeEvents();
    }

    onDrawForeground(ctx) {
        super.onDrawForeground?.(ctx);
        if (!this.executionMessage) {
            return;
        }
        ctx.save();
        ctx.fillStyle = "#fff";
        ctx.font = "16px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(this.executionMessage, this.size[0] / 2, this.size[1] - 16);
        ctx.restore();
    }
}

DirectorActorExecutorNode.title = "Director Actor Executor";
DirectorActorExecutorNode.desc = "Automate Director/Actor group execution";
DirectorActorExecutorNode.type = "Director/ActorExecutor";
DirectorActorExecutorNode.category = "Director/Control";

app.registerExtension({
    name: "DirectorActor.Executor",
    setup() {
        LiteGraph.registerNodeType(DirectorActorExecutorNode.type, DirectorActorExecutorNode);
    },
});
