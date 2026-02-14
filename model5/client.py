import torch
import requests
from transformers import AutoTokenizer

# ----------------------------
# Config
# ----------------------------
SERVER_URL = "http://localhost:8000/forward"
MODEL_NAME = "state-spaces/mamba-2.8b-hf"

# ----------------------------
# Tokenizer (client-side)
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ----------------------------
# Helper: logits → next token
# ----------------------------
def sample_next_token(logits, temperature=1.0, top_k=50):
    logits = logits / temperature

    if top_k is not None:
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values, dim=-1)
        next_id = indices[torch.multinomial(probs, 1)]
    else:
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)

    return next_id.item()

# ----------------------------
# Inference
# ----------------------------
def generate(text, max_new_tokens=20):
    # Tokenize input
    input_ids = tokenizer(
        text,
        return_tensors="pt"
    )["input_ids"]

    input_ids_list = input_ids.tolist()

    for _ in range(max_new_tokens):
        # Send request
        response = requests.post(
            SERVER_URL,
            json={"input_ids": input_ids_list}
        )
        response.raise_for_status()

        logits = torch.tensor(response.json()["logits"])

        # Take last token logits
        last_logits = logits[0, -1]

        # Sample
        next_token_id = sample_next_token(last_logits)

        # Append token
        input_ids_list[0].append(next_token_id)

        # Stop if EOS
        if next_token_id == tokenizer.eos_token_id:
            break

    # Decode
    return tokenizer.decode(input_ids_list[0], skip_special_tokens=True)

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    prompt = "The future of AI is"
    output = generate(prompt)
    print(output)
