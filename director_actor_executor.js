import { app } from "../scripts/app.js";
import { api } from "../scripts/api.js";
import { getGroupOutputNodeIds } from "./director_actor_queue_utils.js";

if (!window.__DAE_state) {
  window.__DAE_state = { active: false, targetIds: [] };
}

let interruptPatched = false;

function patchInterruptFetch() {
  if (interruptPatched) {
    return;
  }
  interruptPatched = true;
  const previousFetch = api.fetchApi.bind(api);
  api.fetchApi = async function (path, options = {}) {
    const response = await previousFetch(path, options);
    if (path === "/interrupt") {
      api.dispatchEvent(new CustomEvent("execution_interrupt"));
    }
    return response;
  };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function randomId() {
  if (window.crypto?.randomUUID) {
    return window.crypto.randomUUID();
  }
  return `dae-${Math.random().toString(36).slice(2, 10)}`;
}

async function waitForQueueIdle(node) {
  while (!node.__cancelling) {
    try {
      const res = await api.fetchApi("/queue");
      const data = await res.json();
      const running = data?.queue_running?.length || 0;
      const pending = data?.queue_pending?.length || 0;
      if (running === 0 && pending === 0) {
        break;
      }
    } catch (error) {
      console.warn("[DirectorActor] queue poll failed", error);
    }
    await sleep(200);
  }
}

async function queueGroup(node, groupName) {
  const outputIds = getGroupOutputNodeIds(groupName);
  if (!outputIds.length) {
    throw new Error(`Group "${groupName}" has no output nodes.`);
  }
  window.__DAE_state.active = true;
  window.__DAE_state.targetIds = outputIds;
  try {
    const nodes = outputIds
      .map((id) => app.graph?.getNodeById?.(id))
      .filter((n) => !!n);
    for (const target of nodes) {
      if (node.__cancelling) {
        break;
      }
      if (typeof target.triggerQueue === "function") {
        await target.triggerQueue();
      } else if (typeof app.queuePrompt === "function") {
        await app.queuePrompt();
      }
      await waitForQueueIdle(node);
      if (node.__cancelling) {
        break;
      }
    }
  } finally {
    window.__DAE_state.active = false;
    window.__DAE_state.targetIds = [];
  }
}

function subscribeDirectorStatus(node) {
  node.__directorListener = (event) => {
    const detail = event?.detail;
    if (!detail || detail.link_id !== node.properties.linkId) {
      return;
    }
    node.__lastDirector = {
      done: Boolean(detail.done),
      prompt: typeof detail.prompt === "string" ? detail.prompt : "",
    };
  };
  api.addEventListener("director-status", node.__directorListener);
}

function unsubscribeDirectorStatus(node) {
  if (node.__directorListener) {
    api.removeEventListener("director-status", node.__directorListener);
    node.__directorListener = null;
  }
}

function subscribeInterrupt(node) {
  node.__interruptListener = () => {
    if (node.__executing) {
      node.__cancelling = true;
    }
  };
  api.addEventListener("execution_interrupt", node.__interruptListener);
}

function unsubscribeInterrupt(node) {
  if (node.__interruptListener) {
    api.removeEventListener("execution_interrupt", node.__interruptListener);
    node.__interruptListener = null;
  }
}

async function executeLoop(node) {
  if (node.__executing) {
    return;
  }
  patchInterruptFetch();
  node.__executing = true;
  node.__cancelling = false;
  node.__lastDirector = null;
  node.properties.iteration = 0;
  subscribeDirectorStatus(node);
  subscribeInterrupt(node);

  try {
    while (!node.__cancelling && node.properties.iteration < node.properties.maxLoops) {
      await queueGroup(node, node.properties.directorGroupName);
      if (node.__cancelling) {
        break;
      }
      const status = node.__lastDirector || {};
      if (status.done || status.prompt === "SUCCESS") {
        break;
      }
      await queueGroup(node, node.properties.actorGroupName);
      if (node.__cancelling) {
        break;
      }
      if (node.properties.delayMsBetweenPhases > 0) {
        await sleep(node.properties.delayMsBetweenPhases);
      }
      node.properties.iteration += 1;
    }
  } catch (error) {
    console.error("[DirectorActor] execution error", error);
  } finally {
    unsubscribeDirectorStatus(node);
    unsubscribeInterrupt(node);
    node.__executing = false;
    node.__cancelling = false;
    window.__DAE_state.active = false;
    window.__DAE_state.targetIds = [];
    node.setDirtyCanvas(true, true);
  }
}

async function cancelExecution(node) {
  if (!node.__executing || node.__cancelling) {
    return;
  }
  node.__cancelling = true;
  try {
    await api.fetchApi("/interrupt", { method: "POST" });
  } catch (error) {
    console.warn("[DirectorActor] cancel failed", error);
  }
}

function updateNodeWidgets(node) {
  node.widgets = node.widgets || [];
  node.widgets.length = 0;
  node.addWidget("text", "Director Group", node.properties.directorGroupName, (value) => {
    node.properties.directorGroupName = value || "";
  });
  node.addWidget("text", "Actor Group", node.properties.actorGroupName, (value) => {
    node.properties.actorGroupName = value || "";
  });
  node.addWidget("number", "Max Loops", node.properties.maxLoops, (value) => {
    node.properties.maxLoops = Math.max(1, Math.floor(value || 1));
  });
  node.addWidget("number", "Delay (ms)", node.properties.delayMsBetweenPhases, (value) => {
    node.properties.delayMsBetweenPhases = Math.max(0, Math.floor(value || 0));
  });
  node.addWidget("text", "Link ID", node.properties.linkId, (value) => {
    node.properties.linkId = value || "";
  });
  node.addWidget("button", "Run", () => executeLoop(node));
  node.addWidget("button", "Cancel", () => cancelExecution(node));
}

app.registerExtension({
  name: "DirectorActorExecutor",
  setup() {
    const NodeClass = function () {
      this.title = "Director Actor Executor";
      this.properties = {
        directorGroupName: "DirectorGroup",
        actorGroupName: "ActorGroup",
        maxLoops: 3,
        delayMsBetweenPhases: 0,
        linkId: randomId(),
        iteration: 0,
      };
      this.size = [260, 240];
      updateNodeWidgets(this);
    };
    NodeClass.title = "DirectorActorExecutor";
    NodeClass.prototype.onConfigure = function () {
      updateNodeWidgets(this);
    };
    NodeClass.prototype.onConnectionsChange = function () {
      updateNodeWidgets(this);
    };
    NodeClass.prototype.onRemoved = function () {
      unsubscribeDirectorStatus(this);
      unsubscribeInterrupt(this);
    };
    LiteGraph.registerNodeType("Director/Actor Executor", NodeClass);
  },
});
