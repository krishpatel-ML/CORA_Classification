"""
GCN Cora API — Production Ready
Optimized for Render free tier (512MB RAM)
- Cora dataset loaded once at startup
- ONNX session loaded once at startup
- Thread limits to reduce memory pressure
- Proper error handling throughout
"""

import os

# ── Memory optimization: limit threads before importing torch/numpy ──────────
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["OPENBLAS_NUM_THREADS"]  = "1"
os.environ["MKL_NUM_THREADS"]       = "1"
os.environ["NUMEXPR_NUM_THREADS"]   = "1"

import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import onnxruntime as ort


# ── Constants ────────────────────────────────────────────────────────────────

CORA_CLASSES = {
    0: "Case_Based",
    1: "Genetic_Algorithms",
    2: "Neural_Networks",
    3: "Probabilistic_Methods",
    4: "Reinforcement_Learning",
    5: "Rule_Learning",
    6: "Theory",
}

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "simple_gcn_cora.onnx")
DATA_DIR   = os.path.join(BASE_DIR, "data", "Planetoid")
STATIC_DIR = os.path.join(BASE_DIR, "static")


# ── Load ONNX model once at startup ─────────────────────────────────────────

print("Loading ONNX model...")
model_session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
print("ONNX model loaded.")


# ── Load Cora dataset once at startup ───────────────────────────────────────
# Previously this was loaded on every /predict/cora_node request,
# which spiked RAM and crashed the free-tier container repeatedly.

print("Loading Cora dataset...")
try:
    from torch_geometric.datasets import Planetoid
    _cora_dataset = Planetoid(root=DATA_DIR, name="Cora")
    cora_data     = _cora_dataset[0]
    CORA_NODE_FEATURES = cora_data.x.numpy()
    CORA_EDGE_INDEX    = cora_data.edge_index.numpy()
    CORA_NUM_NODES     = cora_data.num_nodes
    print(f"Cora dataset loaded. Nodes: {CORA_NUM_NODES}, Edges: {CORA_EDGE_INDEX.shape[1]}")
except Exception as e:
    print(f"WARNING: Could not load Cora dataset: {e}")
    print("The /predict/cora_node endpoint will be unavailable.")
    CORA_NODE_FEATURES = None
    CORA_EDGE_INDEX    = None
    CORA_NUM_NODES     = None


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Simple GCN Cora API",
    description="Graph Convolutional Network for academic paper classification on the Cora dataset.",
    version="1.0.0",
)


# ── Request / Response models ────────────────────────────────────────────────

class GraphPredictRequest(BaseModel):
    node_features: List[List[float]]
    edge_indices: Optional[List[List[int]]] = None


class CoraNodeRequest(BaseModel):
    node_indices: List[int]


# ── Helpers ──────────────────────────────────────────────────────────────────

def softmax(scores: np.ndarray) -> np.ndarray:
    shifted    = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)


def run_model(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    node_indices_to_return: List[int],
) -> dict:
    output = model_session.run(
        ["logits"],
        {
            "node_features": node_features.astype(np.float32),
            "edge_indices":  edge_index.astype(np.int64),
        },
    )
    logits           = output[0]
    probabilities    = softmax(logits)
    predicted_classes = logits.argmax(axis=-1)

    results = []
    for i in node_indices_to_return:
        results.append({
            "node_index":           i,
            "predicted_class_id":   int(predicted_classes[i]),
            "predicted_class_name": CORA_CLASSES[int(predicted_classes[i])],
            "probabilities":        probabilities[i].tolist(),   # fixed spelling
            "logits":               logits[i].tolist(),
        })

    return {
        "num_nodes":   node_features.shape[0],
        "num_edges":   edge_index.shape[1],
        "predictions": results,
    }


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def home_page():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"service": "Simple GCN Cora API", "status": "running"}


@app.get("/health")
def health_check():
    return {
        "status":    "healthy",
        "providers": model_session.get_providers(),
        "cora_dataset_loaded": CORA_NODE_FEATURES is not None,
        "cora_num_nodes":      CORA_NUM_NODES,
    }


@app.get("/info")
def model_info():
    return {
        "Model Name":        "SimpleGCN",
        "feature_dimension": 1433,
        "num_classes":       7,
        "class_mapping":     CORA_CLASSES,
        "inputs": [
            {"name": inp.name, "shape": inp.shape, "type": inp.type}
            for inp in model_session.get_inputs()
        ],
        "outputs": [
            {"name": out.name, "shape": out.shape, "type": out.type}
            for out in model_session.get_outputs()
        ],
    }


@app.post("/predict")
def predict_custom_graph(request: GraphPredictRequest):
    # Validate node features
    if not request.node_features:
        raise HTTPException(400, "node_features cannot be empty.")

    for i, feature_vector in enumerate(request.node_features):
        if len(feature_vector) != 1433:
            raise HTTPException(
                422,
                f"Node {i}: feature vector must have exactly 1433 values, got {len(feature_vector)}."
            )

    node_features = np.array(request.node_features, dtype=np.float32)
    num_nodes     = len(request.node_features)

    # Build or validate edge index
    if request.edge_indices:
        edge_index = np.array(request.edge_indices, dtype=np.int64)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise HTTPException(
                422,
                "edge_indices must have shape [2, num_edges]. "
                "Example: [[0,1],[1,0]] for one bidirectional edge."
            )
        # Check node indices are in bounds
        if edge_index.max() >= num_nodes or edge_index.min() < 0:
            raise HTTPException(
                422,
                f"edge_indices contains out-of-range node index. "
                f"Valid range: 0 to {num_nodes - 1}."
            )
    else:
        # Default: self-loops so every node is connected
        node_ids   = np.arange(num_nodes, dtype=np.int64)
        edge_index = np.vstack([node_ids, node_ids])

    all_node_indices = list(range(num_nodes))
    return run_model(node_features, edge_index, all_node_indices)


@app.post("/predict/cora_node")
def predict_real_cora_nodes(request: CoraNodeRequest):
    # Guard: dataset must have loaded successfully at startup
    if CORA_NODE_FEATURES is None:
        raise HTTPException(
            503,
            "Cora dataset failed to load at startup. "
            "Check server logs for details."
        )

    if not request.node_indices:
        raise HTTPException(400, "node_indices cannot be empty.")

    largest_valid_index = CORA_NUM_NODES - 1
    invalid_indices = [
        i for i in request.node_indices
        if i < 0 or i > largest_valid_index
    ]
    if invalid_indices:
        raise HTTPException(
            400,
            f"node_indices out of bounds: {invalid_indices}. "
            f"Valid range: 0 to {largest_valid_index}."
        )

    return run_model(
        CORA_NODE_FEATURES,
        CORA_EDGE_INDEX,
        request.node_indices,
    )


# ── Static files (must be last) ──────────────────────────────────────────────

if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
