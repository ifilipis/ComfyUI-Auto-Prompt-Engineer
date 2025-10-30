import { app } from "../../scripts/app.js";

const state = {
    active: false,
    targets: new Set(),
};

function activate(targetIds) {
    state.active = Boolean(targetIds?.length);
    state.targets = new Set((targetIds || []).map((id) => String(id)));
}

function deactivate() {
    state.active = false;
    state.targets.clear();
}

function isActive() {
    return state.active && state.targets.size > 0;
}

function getTargets() {
    return new Set(state.targets);
}

function collectRelatedNodes(prompt, nodeId, acc) {
    const key = String(nodeId);
    if (!prompt || !prompt[key] || acc.has(key)) {
        return;
    }
    acc.add(key);
    const node = prompt[key];
    if (!node?.inputs) {
        return;
    }
    Object.values(node.inputs).forEach((input) => {
        if (Array.isArray(input) && input.length > 0) {
            collectRelatedNodes(prompt, input[0], acc);
        }
    });
}

function buildFilteredPrompt(prompt, targets) {
    if (!prompt || !targets?.size) {
        return prompt;
    }
    const relevant = new Set();
    targets.forEach((nodeId) => collectRelatedNodes(prompt, nodeId, relevant));
    const filtered = {};
    relevant.forEach((key) => {
        if (prompt[key]) {
            filtered[key] = prompt[key];
        }
    });
    return filtered;
}

function nodesInGroup(groupName) {
    if (!groupName) {
        return [];
    }
    const group = app.graph?._groups?.find((g) => g?.title === groupName);
    if (!group) {
        return [];
    }
    const nodes = [];
    for (const node of app.graph._nodes) {
        if (!node?.pos || !node?.size) {
            continue;
        }
        if (LiteGraph.overlapBounding(group._bounding, node.getBounding())) {
            nodes.push(node);
        }
    }
    return nodes;
}

function isOutputNode(node) {
    if (!node) {
        return false;
    }
    if (node.constructor?.nodeData?.output_node === true) {
        return true;
    }
    if (!node.outputs || node.outputs.length === 0) {
        return true;
    }
    return node.outputs.every((port) => !port?.links || port.links.length === 0);
}

function outputNodesForGroup(groupName) {
    const nodes = nodesInGroup(groupName);
    return nodes.filter((node) => isOutputNode(node));
}

app.registerExtension({
    name: "DirectorActor.QueueUtils",
    setup() {
        window.__DAE_queueUtils = {
            activate,
            deactivate,
            isActive,
            getTargets,
            buildFilteredPrompt,
            nodesInGroup,
            outputNodesForGroup,
        };
    },
});
