import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MambaForCausalLM, MambaConfig

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Mamba Inference Server")

# ----------------------------
# Load model ONCE
# ----------------------------
MODEL_NAME = "state-spaces/mamba-2.8b-hf"

print("Loading model...")

config = MambaConfig.from_pretrained(
    MODEL_NAME,
    devices=["cuda:0", "cuda:1"]
)

model = MambaForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config
)

# ----------------------------
# Request schema
# ----------------------------
class InferenceRequest(BaseModel):
    input_ids: list[list[int]]  # shape: [batch, seq_len]

# ----------------------------
# Inference endpoint
# ----------------------------
@app.post("/forward")
def forward(req: InferenceRequest):
    input_ids = torch.tensor(
        req.input_ids,
    )

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    return {
        "logits_shape": list(logits.shape),
        "logits": logits.cpu().tolist()
    }
