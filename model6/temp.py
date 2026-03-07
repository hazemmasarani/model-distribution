import os
import torch
import re
import plotly.graph_objects as go
from collections import defaultdict

BASE_DIR = "../model6/layers_output"

MODELS = {
    "orig": os.path.join(BASE_DIR, "orig_mamba"),
    "ssm": os.path.join(BASE_DIR, "ssm_mamba"),
    "gate": os.path.join(BASE_DIR, "gate_mamba"),
}

ATOL = 1e-6
RTOL = 1e-5


def extract_layer_idx(filename):
    match = re.search(r"layer_(\d+)", filename)
    return int(match.group(1)) if match else None


def compare_tensors(t1, t2):
    diff = (t1 - t2).abs()
    return diff.max().item()


def compare_and_collect(modelA_name, modelB_name):
    print(f"\nCollecting stats: {modelA_name} vs {modelB_name}")

    modelA_dir = MODELS[modelA_name]
    modelB_dir = MODELS[modelB_name]

    results = defaultdict(dict)  # results[subfolder][layer_idx] = max_abs_diff

    subfolders = sorted(os.listdir(modelA_dir))

    for sub in subfolders:
        dirA = os.path.join(modelA_dir, sub)
        dirB = os.path.join(modelB_dir, sub)

        if not os.path.exists(dirB):
            continue

        files = sorted(os.listdir(dirA))

        for fname in files:
            layer_idx = extract_layer_idx(fname)
            if layer_idx is None:
                continue

            pathA = os.path.join(dirA, fname)
            pathB = os.path.join(dirB, fname)

            if not os.path.exists(pathB):
                continue

            tA = torch.load(pathA, map_location="cpu")
            tB = torch.load(pathB, map_location="cpu")

            max_diff = compare_tensors(tA, tB)
            results[sub][layer_idx] = max_diff

    return results


def plot_results(results, title):
    fig = go.Figure()

    for subfolder, layer_dict in results.items():
        layers = sorted(layer_dict.keys())
        diffs = [layer_dict[i] for i in layers]

        fig.add_trace(
            go.Scatter(
                x=layers,
                y=diffs,
                mode="lines+markers",
                name=subfolder
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Layer Index",
        yaxis_title="Max Absolute Difference",
        yaxis_type="log",  # important to see small differences
        template="plotly_white",
        height=700,
    )

    fig.show()


# Run comparisons
results_ssm = compare_and_collect("orig", "ssm")
results_gate = compare_and_collect("orig", "gate")

# Plot
plot_results(results_ssm, "Original vs SSM")
plot_results(results_gate, "Original vs Gate")