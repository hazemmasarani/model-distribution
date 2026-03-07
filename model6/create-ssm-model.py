from dotenv import load_dotenv
import os

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json

repo_id = "state-spaces/mamba-2.8b-hf"

# 1) Download index
index_path = hf_hub_download(
    repo_id=repo_id,
    filename="model.safetensors.index.json"
)

with open(index_path, "r") as f:
    index = json.load(f)

# 2) Load all shards
state_dict = {}
for shard_file in sorted(set(index["weight_map"].values())):
    shard_path = hf_hub_download(
        repo_id=repo_id,
        filename=shard_file
    )
    shard_state = load_file(shard_path)
    state_dict.update(shard_state)

# print("Total params loaded:", len(state_dict))
# print(list(state_dict.keys()))


ckpt_keys = list(state_dict.keys())

print("Checkpoint key examples:")
for k in ckpt_keys:
    print(k)

new_state_dict = {}  # use dict, not list
for k in ckpt_keys:
    if "mixer.in_proj.weight" in k:
        # Split rows in half
        rows = state_dict[k].shape[0]
        mid = rows // 2
        new_state_dict[k] = state_dict[k][:mid, :]
    else:
        new_state_dict[k] = state_dict[k]
    
    # new_state_dict[k] = state_dict[k]

print("New checkpoint keys:")
for k in new_state_dict:
    print(k)

from transformers import MambaConfig
from mamba_ssm_modeling import MambaForCausalLM_SSM

config = MambaConfig.from_pretrained("state-spaces/mamba-2.8b-hf",
    ignore_mismatched_sizes=True )
model = MambaForCausalLM_SSM(config)
model.load_state_dict(new_state_dict, strict=False)

# Manually load state dict
model.load_state_dict(new_state_dict, strict=False)  # strict=False because some keys were removed


from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

model.push_to_hub(
    repo_id="hmasarani/mamba-ssm",  # new repo
    private=False
)