import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import "./director_actor_prompt_filter.js";

const daeState = window.__DAE_state || (window.__DAE_state = {});

daeState.targetIds = daeState.targetIds || new Set();
daeState.executors = daeState.executors || new Set();
daeState.activeCount = daeState.activeCount || 0;
daeState.active = daeState.active || false;
daeState.currentLinkId = daeState.currentLinkId || null;
daeState.phase = daeState.phase || null;
daeState.phaseGroupName = daeState.phaseGroupName || null;
daeState.phaseTargetIds = daeState.phaseTargetIds || new Set();
daeState.phaseTargetSummary = daeState.phaseTargetSummary || [];
daeState.directorOutputId = daeState.directorOutputId || null;
daeState.stopOnSuccess = daeState.stopOnSuccess || false;

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

function overlapRect(a, b) {
  const [ax, ay, aw, ah] = a;
  const [bx, by, bw, bh] = b;
  return ax < bx + bw && ax + aw > bx && ay < by + bh && ay + ah > by;
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

function updateTargetIds(ids) {
  daeState.targetIds.clear();
  ids.forEach((id) => daeState.targetIds.add(id));
}

function rectContainsRect(container, rect) {
  if (!Array.isArray(container) || !Array.isArray(rect)) {
    return false;
  }
  const [cx, cy, cw, ch] = container;
  const [rx, ry, rw, rh] = rect;
  return (
    rx >= cx &&
    ry >= cy &&
    rx + (rw || 0) <= cx + cw &&
    ry + (rh || 0) <= cy + ch
  );
}

function isNodeInsideGroupStrict(node, group) {
  if (!node || !group) {
    return false;
  }

  if (typeof group.isNodeInside === "function") {
    try {
      if (group.isNodeInside(node)) {
        return true;
      }
    } catch (error) {
      console.warn("[DirectorActor] group.isNodeInside failed", error);
    }
  }

  const groupRect = getRect(group);
  const nodeRect = getRect(node);

  if (rectContainsRect(groupRect, nodeRect)) {
    return true;
  }

  if (typeof group.isPointInside === "function") {
    const [nx, ny, nw, nh] = nodeRect;
    const corners = [
      [nx, ny],
      [nx + (nw || 0), ny],
      [nx, ny + (nh || 0)],
      [nx + (nw || 0), ny + (nh || 0)],
    ];
    try {
      return corners.every(([x, y]) => group.isPointInside(x, y, 0));
    } catch (error) {
      console.warn("[DirectorActor] group.isPointInside failed", error);
    }
  }

  return false;
}

function getNodeGroupTitles(node) {
  const groups = app.graph?._groups || [];
  const titles = [];
  for (const group of groups) {
    if (!group) {
      continue;
    }
    try {
      if (isNodeInsideGroupStrict(node, group)) {
        titles.push(group?.title || "");
      }
    } catch (error) {
      console.warn("[DirectorActor] Failed to inspect group membership", error);
    }
  }
  return titles;
}

function setPhaseContext(phase, groupName, nodes) {
  daeState.phase = phase;
  daeState.phaseGroupName = groupName;
  if (!daeState.phaseTargetIds) {
    daeState.phaseTargetIds = new Set();
  } else {
    daeState.phaseTargetIds.clear();
  }

  const numericIds = [];
  const audit = [];
  for (const node of nodes || []) {
    if (!node || node.id == null) {
      continue;
    }
    numericIds.push(node.id);
    daeState.phaseTargetIds.add(String(node.id));
    audit.push({
      id: node.id,
      title: node.title,
      groups: getNodeGroupTitles(node),
    });
  }

  if (phase === "director") {
    daeState.directorOutputId = audit.length ? String(audit[0].id) : null;
  }

  daeState.phaseTargetSummary = audit;
  updateTargetIds(numericIds);

  debugLog("Phase target audit", {
    phase,
    group: groupName,
    targets: audit,
  });
}

function clearPhaseContext() {
  daeState.phase = null;
  daeState.phaseGroupName = null;
  daeState.directorOutputId = null;
  daeState.phaseTargetSummary = [];
  if (daeState.phaseTargetIds) {
    daeState.phaseTargetIds.clear();
  } else {
    daeState.phaseTargetIds = new Set();
  }
}

function debugLog(message, payload) {
  console.debug(`[DirectorActor] ${message}`, payload || "");
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
    this.stopGateEngaged = false;
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

    clearPhaseContext();
    daeState.stopOnSuccess = false;
    this.stopGateEngaged = false;

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
      await this.queueGroupAndWait(this.properties.directorGroupName);
      if (this.properties.isCancelling) {
        break;
      }

      debugLog("Awaiting director status", { iteration });
      const directorStatus = await this.waitForDirectorSignal(undefined, iteration);
      const resolvedStatus = directorStatus || this.lastDirectorStatus || { done: false, prompt: "" };
      debugLog("Director status received", resolvedStatus);
      const actorDecision = this.evaluateDirectorStatus(resolvedStatus, iteration);
      if (!actorDecision.runActor) {
        if (this.stopGateEngaged) {
          await this.interruptIfQueueBusy();
        }
        this.properties.iter = iteration + 1;
        this.updateStatus(actorDecision.statusText);
        break;
      }

      debugLog("Queueing actor group", { iteration });
      this.updateStatus(`Actor pass ${iteration + 1}`);
      await this.queueGroupAndWait(this.properties.actorGroupName);
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
      this.engageStopGate("done");
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
      this.engageStopGate("success");
      return {
        runActor: false,
        statusText: "Director SUCCESS",
      };
    }

    return defaultDecision;
  }

  async queueGroupAndWait(groupName) {
    const outputNodes = this.getGroupOutputNodes(groupName);
    if (!outputNodes.length) {
      throw new Error(`No output nodes found in group "${groupName}"`);
    }

    const phase = this.resolvePhase(groupName);
    const phaseNodes = this.normalizePhaseTargets(phase, outputNodes, groupName);
    if (!phaseNodes.length) {
      throw new Error(`No valid output targets resolved for group "${groupName}"`);
    }

    daeState.currentLinkId = this.ensureLinkId();
    setPhaseContext(phase, groupName, phaseNodes);
    const nodeIds = phaseNodes.map((node) => node.id);

    try {
      const needsFinalWait = await this.queueNodes(nodeIds, phaseNodes, phase);
      if (needsFinalWait) {
        await this.waitForQueueIdle();
      }
    } finally {
      daeState.targetIds.clear();
      clearPhaseContext();
    }
  }

  async queueNodes(nodeIds, nodes, phase) {
    if (this.stopGateEngaged || daeState.stopOnSuccess) {
      debugLog("Stop gate active, skipping queue dispatch", {
        phase,
        nodeIds,
      });
      return false;
    }

    if (window.rgthree?.queueOutputNodes) {
      debugLog("Using rgthree queueOutputNodes", { phase, nodeIds });
      await window.rgthree.queueOutputNodes(nodeIds);
      return true;
    }

    const allowedIds = new Set((nodes || []).map((node) => node?.id));
    const triggerable = [];
    const fallback = [];

    for (const node of nodes) {
      if (!allowedIds.has(node?.id)) {
        debugLog("Skipping node outside phase targets", { phase, node: node?.id });
        continue;
      }
      if (typeof node.triggerQueue === "function") {
        triggerable.push(node);
      } else {
        fallback.push(node);
      }
    }

    for (const node of triggerable) {
      if (this.properties.isCancelling) {
        break;
      }
      debugLog("Triggering node queue", { node: node.id, phase });
      await node.triggerQueue();
      await this.waitForQueueIdle();
    }

    if (this.properties.isCancelling) {
      return false;
    }

    if (fallback.length > 0) {
      const fallbackIds = fallback.map((node) => node.id);
      debugLog("Falling back to app.queuePrompt", { phase, nodes: fallbackIds });
      if (daeState.stopOnSuccess) {
        debugLog("Stop gate prevented fallback queue", { nodes: fallbackIds });
        return false;
      }
      await app.queuePrompt();
      await this.waitForQueueIdle();
    }

    return false;
  }

  resolvePhase(groupName) {
    if (groupName === this.properties.directorGroupName) {
      return "director";
    }
    if (groupName === this.properties.actorGroupName) {
      return "actor";
    }
    return "unknown";
  }

  normalizePhaseTargets(phase, nodes, groupName) {
    const available = Array.isArray(nodes) ? nodes.filter(Boolean) : [];
    if (phase === "director" && available.length > 1) {
      const preferred =
        available.find((node) =>
          String(node?.title || "").toLowerCase().includes("director")
        ) || available[0];
      debugLog("Director phase limited to single output", {
        group: groupName,
        preferred: preferred?.id,
        candidates: available.map((node) => node?.id),
      });
      return preferred ? [preferred] : [];
    }
    return available;
  }

  engageStopGate(reason) {
    if (this.stopGateEngaged) {
      return;
    }
    this.stopGateEngaged = true;
    daeState.stopOnSuccess = true;
    debugLog("Stop gate engaged", { reason });
  }

  async interruptIfQueueBusy() {
    try {
      const response = await api.fetchApi("/queue");
      if (!response?.ok) {
        return;
      }
      const data = await response.json();
      const running = data?.queue_running?.length || 0;
      const pending = data?.queue_pending?.length || 0;
      if (running > 0 || pending > 0) {
        debugLog("Stop gate interrupting active queue", { running, pending });
        try {
          await api.fetchApi("/interrupt", { method: "POST" });
        } catch (error) {
          console.warn("[DirectorActor] Stop gate interrupt failed", error);
        }
      }
    } catch (error) {
      console.warn("[DirectorActor] Failed to inspect queue for stop gate", error);
    }
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

  getGroupOutputNodes(groupName) {
    const group = app.graph?._groups?.find((g) => g?.title === groupName);
    if (!group) {
      throw new Error(`Group "${groupName}" not found`);
    }

    const nodesInGroup = [];
    for (const node of app.graph._nodes || []) {
      if (!node) {
        continue;
      }
      if (isNodeInsideGroupStrict(node, group)) {
        nodesInGroup.push(node);
      }
    }

    return nodesInGroup.filter((node) => this.isOutputNode(node));
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
    clearPhaseContext();
    daeState.stopOnSuccess = false;
    this.stopGateEngaged = false;
    daeState.executors.delete(this);
    daeState.activeCount = Math.max(0, daeState.activeCount - 1);
    if (daeState.activeCount === 0) {
      daeState.active = false;
      daeState.targetIds.clear();
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
