import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import "./director_actor_prompt_filter.js";

const daeState = window.__DAE_state || (window.__DAE_state = {});

daeState.targetIds = daeState.targetIds || new Set();
daeState.allowedTargetIds = daeState.allowedTargetIds || new Set();
daeState.executors = daeState.executors || new Set();
daeState.activeCount = daeState.activeCount || 0;
daeState.active = daeState.active || false;
daeState.currentLinkId = daeState.currentLinkId || null;
daeState.phase = daeState.phase || null;
daeState.phaseGroupName = daeState.phaseGroupName || null;
daeState.stopRequested = daeState.stopRequested || false;

defineInterruptHooks();

function defineInterruptHooks() {
  if (daeState.__interruptPatched) {
    return;
  }

  const originalFetchApi = api.fetchApi.bind(api);
  api.fetchApi = async function patchedFetchApi(path, options = {}) {
    const response = await originalFetchApi(path, options);
    if (path === "/interrupt") {
      queueMicrotask(() => {
        api.dispatchEvent(new CustomEvent("execution_interrupt"));
      });
    }
    return response;
  };

  const interruptListener = () => {
    daeState.executors.forEach((node) => {
      node?.handleInterrupt?.();
    });
  };
  api.addEventListener("execution_interrupt", interruptListener);

  daeState.__interruptPatched = true;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function generateLinkId() {
  if (window.crypto?.randomUUID) {
    return window.crypto.randomUUID();
  }
  return `dae-${Math.random().toString(16).slice(2, 10)}-${Date.now()}`;
}

function getRect(entity) {
  if (!entity) {
    return [0, 0, 0, 0];
  }
  if (typeof entity.getBounding === "function") {
    return entity.getBounding();
  }
  const pos = entity.pos || [0, 0];
  const size = entity.size || [0, 0];
  return [pos[0], pos[1], size[0], size[1]];
}

function rectContains(container, inner) {
  const [cx, cy, cw, ch] = container;
  const [ix, iy, iw, ih] = inner;
  if (cw <= 0 || ch <= 0) {
    return false;
  }
  const insideX = ix >= cx && ix + iw <= cx + cw;
  const insideY = iy >= cy && iy + ih <= cy + ch;
  return insideX && insideY;
}

function ensureNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function ensureInteger(value, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  const truncated = Math.trunc(parsed);
  return Number.isFinite(truncated) ? truncated : fallback;
}

function debugLog(message, payload) {
  console.debug(`[DirectorActor] ${message}`, payload || "");
}

function replaceSetContents(targetSet, values) {
  targetSet.clear();
  for (const value of values || []) {
    targetSet.add(value);
  }
}

function setPhaseTargets(phase, groupName, targetIds) {
  replaceSetContents(daeState.targetIds, targetIds);
  replaceSetContents(daeState.allowedTargetIds, targetIds);
  daeState.phase = phase || null;
  daeState.phaseGroupName = groupName || null;
  daeState.stopRequested = false;
}

function clearPhaseTargets() {
  daeState.phase = null;
  daeState.phaseGroupName = null;
  daeState.targetIds.clear();
  daeState.allowedTargetIds.clear();
}

function findNodeById(nodeId) {
  if (!app?.graph) {
    return null;
  }
  if (typeof app.graph.getNodeById === "function") {
    return app.graph.getNodeById(nodeId) || null;
  }
  const nodes = app.graph._nodes || [];
  for (const node of nodes) {
    if (node?.id === nodeId) {
      return node;
    }
  }
  return null;
}

function findGroupTitleForNode(node) {
  if (!node) {
    return null;
  }
  const nodeRect = getRect(node);
  const groups = app?.graph?._groups || [];
  for (const group of groups) {
    if (rectContains(getRect(group), nodeRect)) {
      return group?.title || null;
    }
  }
  return null;
}

class DirectorActorExecutorNode extends LiteGraph.LGraphNode {
  constructor() {
    super();
    this.title = "DirectorActorExecutor";
    this.properties = {
      directorGroupName: "DirectorGroup",
      actorGroupName: "ActorGroup",
      maxLoops: 5,
      delayMsBetweenPhases: 0,
      linkId: generateLinkId(),
      isExecuting: false,
      isCancelling: false,
      iter: 0,
      status: "Idle",
    };
    this.lastDirectorStatus = null;
    this.size = [320, 180];
    this.maxLoopsWidget = this.addWidget(
      "number",
      "Max loops",
      this.properties.maxLoops,
      (value) => this.setMaxLoops(value),
      { min: 1, step: 1, precision: 0 }
    );
    if (this.maxLoopsWidget) {
      this.maxLoopsWidget.options = {
        ...this.maxLoopsWidget.options,
        min: 1,
        step: 1,
        precision: 0,
      };
      this.maxLoopsWidget.serializeValue = (value) =>
        Math.max(1, ensureInteger(value, 1));
    }
    this.setMaxLoops(this.properties.maxLoops);
    this.addWidget("button", "Run", null, () => this.startExecution());
    this.addWidget("button", "Cancel", null, () => this.cancelExecution());
  }

  setMaxLoops(value) {
    const sanitized = Math.max(1, ensureInteger(value, 1));
    this.properties.maxLoops = sanitized;
    if (this.maxLoopsWidget) {
      this.maxLoopsWidget.value = sanitized;
    }
  }

  onPropertyChanged(name, value) {
    if (name === "maxLoops") {
      this.setMaxLoops(value);
    }
  }

  onDrawForeground(ctx) {
    super.onDrawForeground?.(ctx);
    if (this.flags.collapsed) {
      return;
    }
    ctx.save();
    ctx.font = "16px sans-serif";
    ctx.textAlign = "left";
    ctx.fillStyle = this.properties.isExecuting ? "dodgerblue" : "#999";
    ctx.fillText(`Status: ${this.properties.status}`, 10, this.size[1] - 24);
    ctx.fillText(`Iter: ${this.properties.iter}`, 10, this.size[1] - 8);
    ctx.restore();
  }

  async startExecution() {
    if (this.properties.isExecuting) {
      return;
    }

    clearPhaseTargets();
    daeState.stopRequested = false;
    this.resetSessionLink();
    debugLog("Run requested", { linkId: this.properties.linkId });

    this.properties.isExecuting = true;
    this.properties.isCancelling = false;
    this.properties.iter = 0;
    this.lastDirectorStatus = null;
    this.updateStatus("Starting loop");
    daeState.executors.add(this);
    daeState.activeCount += 1;
    daeState.active = true;
    this.attachEventListeners();

    try {
      await this.executeLoop();
    } catch (error) {
      console.error("[DirectorActor] Execution failed", error);
      app.ui?.dialog?.show?.(`DirectorActorExecutor error: ${error.message || error}`);
    } finally {
      this.cleanupAfterExecution();
    }
  }

  async executeLoop() {
    const maxLoops = Math.max(1, ensureInteger(this.properties.maxLoops, 1));
    let iteration = 0;

    while (!this.properties.isCancelling && iteration < maxLoops) {
      this.lastDirectorStatus = null;
      debugLog("Queueing director group", { iteration });
      this.updateStatus(`Director pass ${iteration + 1}`);
      await this.queueGroupAndWait(this.properties.directorGroupName, "director");
      if (this.properties.isCancelling) {
        break;
      }

      debugLog("Awaiting director status", { iteration });
      const directorStatus = await this.waitForDirectorSignal(undefined, iteration);
      const resolvedStatus = directorStatus || this.lastDirectorStatus || { done: false, prompt: "" };
      debugLog("Director status received", resolvedStatus);
      const actorDecision = this.evaluateDirectorStatus(resolvedStatus, iteration);
      if (!actorDecision.runActor) {
        this.properties.iter = iteration + 1;
        this.updateStatus(actorDecision.statusText);
        await this.handleDirectorStop(actorDecision, resolvedStatus, iteration);
        break;
      }

      debugLog("Queueing actor group", { iteration });
      this.updateStatus(`Actor pass ${iteration + 1}`);
      await this.queueGroupAndWait(this.properties.actorGroupName, "actor");
      if (this.properties.isCancelling) {
        break;
      }

      iteration += 1;
      this.properties.iter = iteration;

      const delayMs = Math.max(0, ensureNumber(this.properties.delayMsBetweenPhases, 0));
      if (delayMs > 0 && !this.properties.isCancelling) {
        debugLog("Sleeping between phases", { delayMs });
        await sleep(delayMs);
      }
    }
  }

  resetSessionLink() {
    const previousLinkId = this.properties.linkId;
    this.properties.linkId = generateLinkId();
    debugLog("Session link reset", {
      previous: previousLinkId,
      next: this.properties.linkId,
    });
  }

  async waitForDirectorSignal(timeoutMs = 60000, iteration = null) {
    const start = Date.now();
    while (!this.properties.isCancelling) {
      if (this.lastDirectorStatus) {
        return this.lastDirectorStatus;
      }
      if (timeoutMs != null && Date.now() - start >= timeoutMs) {
        debugLog("Director status wait timed out", { timeoutMs, iteration });
        return this.lastDirectorStatus;
      }
      await sleep(50);
    }
    return this.lastDirectorStatus;
  }

  evaluateDirectorStatus(status, iteration) {
    const defaultDecision = { runActor: true, statusText: "Director awaiting actor" };
    if (!status) {
      debugLog("Director status missing, continuing to actor", { iteration });
      return defaultDecision;
    }

    if (status.done === true) {
      debugLog("Director reported done, skipping actor", { iteration, status });
      return {
        runActor: false,
        statusText: "Director completed",
      };
    }

    const promptText = typeof status.prompt === "string" ? status.prompt.trim() : "";
    if (promptText.toUpperCase() === "SUCCESS") {
      debugLog("Director returned SUCCESS, skipping actor", {
        iteration,
        prompt: promptText,
      });
      return {
        runActor: false,
        statusText: "Director SUCCESS",
      };
    }

    return defaultDecision;
  }

  async handleDirectorStop(decision, status, iteration) {
    daeState.stopRequested = true;
    daeState.active = false;
    clearPhaseTargets();
    debugLog("Director stop requested", {
      decision,
      status,
      iteration,
    });

    try {
      const response = await api.fetchApi("/queue");
      if (!response?.ok) {
        throw new Error(`Queue status ${response?.status}`);
      }
      const data = await response.json();
      const running = data?.queue_running?.length || 0;
      const pending = data?.queue_pending?.length || 0;
      if (running > 0 || pending > 0) {
        debugLog("Interrupting trailing queue after stop", { running, pending });
        await api.fetchApi("/interrupt", { method: "POST" });
      }
    } catch (error) {
      console.warn("[DirectorActor] Failed to stop queue after director stop", error);
    }
  }

  async queueGroupAndWait(groupName, phase) {
    const { group, outputNodes } = this.getGroupContext(groupName);
    if (!outputNodes.length) {
      throw new Error(`No output nodes found in group "${groupName}"`);
    }

    const groupTitle = group?.title || groupName;
    const filteredOutputs = outputNodes.filter((node) => {
      const nodeGroup = findGroupTitleForNode(node);
      const isMatch = nodeGroup === groupTitle;
      if (!isMatch) {
        debugLog("Dropping output outside group bounds", {
          node: node?.id,
          nodeGroup,
          expectedGroup: groupTitle,
        });
      }
      return isMatch;
    });

    if (!filteredOutputs.length) {
      throw new Error(`No valid outputs resolved for group "${groupTitle}"`);
    }
    if (filteredOutputs.length !== outputNodes.length) {
      debugLog("Filtered outputs to enforce group bounds", {
        group: groupTitle,
        before: outputNodes.map((node) => node?.id),
        after: filteredOutputs.map((node) => node?.id),
      });
    }

    daeState.currentLinkId = this.ensureLinkId();
    const nodeIds = filteredOutputs.map((node) => node.id);
    setPhaseTargets(phase, groupTitle, nodeIds);

    const targetSummaries = nodeIds.map((id) => {
      const node = findNodeById(id);
      return {
        id,
        group: findGroupTitleForNode(node),
        title: node?.title || null,
      };
    });
    debugLog("Prepared phase targets", {
      phase,
      group: group?.title || groupName,
      targets: targetSummaries,
    });

    try {
      const needsFinalWait = await this.queueNodes(nodeIds, filteredOutputs, phase);
      if (needsFinalWait) {
        await this.waitForQueueIdle();
      }
    } finally {
      clearPhaseTargets();
    }
  }

  async queueNodes(nodeIds, nodes, phase) {
    if (window.rgthree?.queueOutputNodes) {
      debugLog("Using rgthree queueOutputNodes", nodeIds);
      await window.rgthree.queueOutputNodes(nodeIds);
      return true;
    }

    for (const node of nodes) {
      if (this.properties.isCancelling) {
        break;
      }
      if (daeState.stopRequested) {
        debugLog("Stop requested before queue trigger", { phase, node: node.id });
        break;
      }
      if (typeof node.triggerQueue === "function") {
        debugLog("Triggering node queue", { node: node.id });
        await node.triggerQueue();
        await this.waitForQueueIdle();
      } else {
        if (phase === "director") {
          debugLog("Skipping fallback enqueue for director phase", { node: node.id });
          continue;
        }
        debugLog("Falling back to app.queuePrompt", { node: node.id });
        await app.queuePrompt();
        await this.waitForQueueIdle();
      }
    }

    return false;
  }

  async waitForQueueIdle() {
    while (!this.properties.isCancelling) {
      try {
        const response = await api.fetchApi("/queue");
        if (!response?.ok) {
          throw new Error(`Queue status ${response?.status}`);
        }
        const data = await response.json();
        const running = data?.queue_running?.length || 0;
        const pending = data?.queue_pending?.length || 0;
        debugLog("Queue poll", { running, pending });
        if (running === 0 && pending === 0) {
          return;
        }
      } catch (error) {
        console.warn("[DirectorActor] Queue polling error", error);
      }
      await sleep(200);
    }
  }

  ensureLinkId() {
    if (!this.properties.linkId) {
      this.properties.linkId = generateLinkId();
    }
    return this.properties.linkId;
  }

  getGroupContext(groupName) {
    const group = app.graph?._groups?.find((g) => g?.title === groupName);
    if (!group) {
      throw new Error(`Group "${groupName}" not found`);
    }

    const groupRect = getRect(group);
    const nodesInGroup = [];
    for (const node of app.graph._nodes || []) {
      if (!node || !node.pos) {
        continue;
      }
      if (rectContains(groupRect, getRect(node))) {
        nodesInGroup.push(node);
      }
    }

    const outputNodes = nodesInGroup.filter((node) => this.isOutputNode(node));
    return { group, groupRect, outputNodes };
  }

  isOutputNode(node) {
    if (!node || node.mode === LiteGraph.NEVER) {
      return false;
    }
    if (node?.constructor?.nodeData?.output_node) {
      return true;
    }
    if (!node?.outputs || node.outputs.length === 0) {
      return true;
    }
    return node.outputs.every((output) => !output?.links || output.links.length === 0);
  }

  attachEventListeners() {
    this.directorHandler = (event) => {
      const detail = event?.detail;
      if (!detail || detail.link_id !== this.properties.linkId) {
        return;
      }
      this.lastDirectorStatus = {
        done: Boolean(detail.done),
        prompt: detail.prompt ?? "",
      };
      debugLog("Received director-status event", this.lastDirectorStatus);
    };

    this.imageHandler = (event) => {
      const detail = event?.detail;
      if (!detail || detail.link_id !== this.properties.linkId) {
        return;
      }
      debugLog("Received img-ready event", detail);
    };

    api.addEventListener("director-status", this.directorHandler);
    api.addEventListener("img-ready", this.imageHandler);
  }

  detachEventListeners() {
    if (this.directorHandler) {
      api.removeEventListener("director-status", this.directorHandler);
      this.directorHandler = null;
    }
    if (this.imageHandler) {
      api.removeEventListener("img-ready", this.imageHandler);
      this.imageHandler = null;
    }
  }

  updateStatus(text) {
    this.properties.status = text;
    this.setDirtyCanvas(true, true);
  }

  handleInterrupt() {
    if (this.properties.isExecuting && !this.properties.isCancelling) {
      debugLog("Interrupt event received", { linkId: this.properties.linkId });
      this.properties.isCancelling = true;
      this.updateStatus("Cancelling");
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
    this.updateStatus("Cancelling");

    try {
      await api.fetchApi("/interrupt", { method: "POST" });
    } catch (error) {
      console.error("[DirectorActor] Cancel request failed", error);
    }
  }

  cleanupAfterExecution() {
    this.detachEventListeners();
    daeState.executors.delete(this);
    daeState.activeCount = Math.max(0, daeState.activeCount - 1);
    if (daeState.activeCount === 0) {
      daeState.active = false;
      clearPhaseTargets();
      daeState.stopRequested = false;
    }
    if (daeState.currentLinkId === this.properties.linkId) {
      daeState.currentLinkId = null;
    }
    this.properties.isExecuting = false;
    this.properties.isCancelling = false;
    this.updateStatus("Idle");
    this.setDirtyCanvas(true, true);
  }
}

app.registerExtension({
  name: "DirectorActorExecutor",
  registerCustomNodes() {
    LiteGraph.registerNodeType("director/DirectorActorExecutor", DirectorActorExecutorNode);
  },
});
