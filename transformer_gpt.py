
import math
import ssl
from urllib.request import urlopen

import certifi
import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================
# 1. HYPERPARAMETERS
# ============================================================

batch_size = 32
block_size = 128          # Maximum context length
max_iters = 1000
eval_interval = 300
eval_iters = 50
learning_rate = 3e-4

n_embd = 96               # Embedding dimension
n_head = 4                # Number of attention heads
n_layer = 3               # Number of Transformer blocks
dropout = 0.1

torch.manual_seed(1337)

if torch.cuda.is_available():
    device = "cuda"
elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)


# ============================================================
# 2. LOAD DATASET
# ============================================================

url = (
    "https://raw.githubusercontent.com/MandishaShivam/nanogpt-model/main/master/data/conversation-input.txt"
)

try:
    ssl_context = ssl.create_default_context(
        cafile=certifi.where()
    )

    with urlopen(
        url,
        timeout=30,
        context=ssl_context
    ) as response:
        text = response.read().decode("utf-8")

except Exception as e:
    raise RuntimeError(
        f"Could not download dataset: {e}"
    ) from e

print("Dataset loaded successfully.")
print("Total characters:", len(text))
print("Dataset preview:\n", text[:300])


# ============================================================
# 3. CHARACTER TOKENIZER
# ============================================================

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[int(i)] for i in ids)


print("Vocabulary size:", vocab_size)


# ============================================================
# 4. TRAIN / VALIDATION SPLIT
# ============================================================

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

if len(train_data) <= block_size or len(val_data) <= block_size:
    raise ValueError("Dataset is too short for block_size.")

print("Training characters:", len(train_data))
print("Validation characters:", len(val_data))


# ============================================================
# 5. RANDOM BATCH GENERATION
# ============================================================

def get_batch(split):
    source = train_data if split == "train" else val_data

    ix = torch.randint(
        len(source) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        source[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        source[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)


# ============================================================
# 6. CAUSAL SELF-ATTENTION HEAD
# ============================================================

class Head(nn.Module):
    """One self-attention head."""

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Prevent attention to future characters.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        # Attention scores: (B, T, T)
        weights = q @ k.transpose(-2, -1)
        weights = weights * (k.shape[-1] ** -0.5)

        # Mask future positions.
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)

        return weights @ v


# ============================================================
# 7. MULTI-HEAD SELF-ATTENTION
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(
            num_heads * head_size,
            n_embd
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        return self.dropout(self.proj(out))


# ============================================================
# 8. FEED-FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 9. TRANSFORMER BLOCK
# ============================================================

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(
            n_head,
            head_size
        )

        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual connections + pre-layer normalization.
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


# ============================================================
# 10. GPT LANGUAGE MODEL
# ============================================================

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)

        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(positions)

        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits_flat = logits.reshape(B * T, C)
            targets_flat = targets.reshape(B * T)

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            # Keep only the most recent context.
            idx_cond = idx[:, -block_size:]

            logits, _ = self(idx_cond)

            # Use the final time step for next-character prediction.
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx


# ============================================================
# 11. INITIALIZE MODEL AND OPTIMIZER
# ============================================================

model = GPTLanguageModel().to(device)

num_params = sum(
    p.numel() for p in model.parameters()
)

print(f"\nModel parameters: {num_params:,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# 12. ESTIMATE LOSS
# ============================================================

@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)

            _, loss = model(X, Y)
            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()
    return results


# ============================================================
# 13. TRAIN THE TRANSFORMER
# ============================================================

print("\nStarting Transformer training...\n")

for iteration in range(max_iters):

    if iteration % eval_interval == 0:
        losses = estimate_loss()

        print(
            f"Step {iteration:4d} | "
            f"Train loss: {losses['train']:.4f} | "
            f"Validation loss: {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Avoid unusually large gradient updates.
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()


# Final evaluation
losses = estimate_loss()

print("\nTraining completed.")
print(f"Final train loss: {losses['train']:.4f}")
print(f"Final validation loss: {losses['val']:.4f}")


# ============================================================
# 14. SAVE MODEL CHECKPOINT
# ============================================================

checkpoint = {
    "model_state_dict": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "vocab_size": vocab_size,
    "block_size": block_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout,
}

torch.save(checkpoint, "conversation_transformer.pt")

print("\nModel saved to conversation_transformer.pt")


# ============================================================
# 15. GENERATE TEXT
# ============================================================

model.eval()

prompt = "Human 1:"

if any(ch not in stoi for ch in prompt):
    raise ValueError(
        "Prompt contains characters absent from the vocabulary."
    )

context = torch.tensor(
    [encode(prompt)],
    dtype=torch.long,
    device=device
)

with torch.no_grad():
    generated_ids = model.generate(
        context,
        max_new_tokens=500
    )[0].tolist()

generated_text = decode(generated_ids)

print("\n" + "=" * 60)
print("GENERATED TRANSFORMER TEXT")
print("=" * 60)
print(generated_text)


# ============================================================
# 16. SAVE GENERATED TEXT
# ============================================================

with open(
    "generated_transformer.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(generated_text)

print("\nGenerated text saved to generated_transformer.txt")
