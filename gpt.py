
import torch
import torch.nn as nn
from torch.nn import functional as F
from urllib.request import urlopen

# ============================================================
# 1. HYPERPARAMETERS
# ============================================================

batch_size = 16
block_size = 128

max_iters = 2000
eval_interval = 100
eval_iters = 50

learning_rate = 3e-4

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1

# Device selection: NVIDIA GPU, Apple Silicon GPU, or CPU
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(1337)

print("Device:", device)


# ============================================================
# 2. LOAD YOUR CONVERSATION DATASET
# ============================================================

url = (
    "https://raw.githubusercontent.com/"
    "MandishaShivam/nanogpt-model/main/master/data/"
    "conversation-input.txt"
)

with urlopen(url, timeout=30) as response:
    text = response.read().decode("utf-8")

print("Dataset loaded successfully.")
print("Total characters:", len(text))
print("\nDataset preview:\n")
print(text[:500])


# ============================================================
# 3. BUILD THE CHARACTER-LEVEL TOKENIZER
# ============================================================

chars = sorted(list(set(text)))
vocab_size = len(chars)

# Character -> integer ID
stoi = {ch: i for i, ch in enumerate(chars)}

# Integer ID -> character
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    """Convert a string into a list of integer IDs."""
    return [stoi[c] for c in s]

def decode(ids):
    """Convert integer IDs back into a string."""
    return "".join([itos[int(i)] for i in ids])

print("\nVocabulary size:", vocab_size)


# ============================================================
# 4. CONVERT TEXT TO A TENSOR AND SPLIT THE DATA
# ============================================================

data = torch.tensor(encode(text), dtype=torch.long)

# First 90% for training, remaining 10% for validation
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

print("Training characters:", len(train_data))
print("Validation characters:", len(val_data))

if len(train_data) <= block_size or len(val_data) <= block_size:
    raise ValueError(
        "The dataset must contain more than block_size "
        "characters in both training and validation splits."
    )


# ============================================================
# 5. CREATE TRAINING BATCHES
# ============================================================

def get_batch(split):
    """
    Randomly sample batches of input sequences x and targets y.
    Each target sequence is shifted one character forward.
    """

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

    x = x.to(device)
    y = y.to(device)

    return x, y


# ============================================================
# 6. ESTIMATE TRAINING AND VALIDATION LOSS
# ============================================================

@torch.no_grad()
def estimate_loss():
    out = {}

    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()

    return out


# ============================================================
# 7. SINGLE SELF-ATTENTION HEAD
# ============================================================

class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Prevent attention to future tokens
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        # Attention scores
        wei = q @ k.transpose(-2, -1)
        wei = wei * (k.shape[-1] ** -0.5)

        # Causal mask
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Weighted sum of values
        v = self.value(x)
        out = wei @ v

        return out


# ============================================================
# 8. MULTI-HEAD SELF-ATTENTION
# ============================================================

class MultiHeadAttention(nn.Module):
    """Multiple self-attention heads operating in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        out = self.dropout(self.proj(out))

        return out


# ============================================================
# 9. FEED-FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):
    """Position-wise feed-forward neural network."""

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 10. TRANSFORMER BLOCK
# ============================================================

class Block(nn.Module):
    """Communication through attention, followed by computation."""

    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


# ============================================================
# 11. GPT LANGUAGE MODEL
# ============================================================

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Character embeddings
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd
        )

        # Position embeddings
        self.position_embedding_table = nn.Embedding(
            block_size, n_embd
        )

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)

        # Predict the next character
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token and positional embeddings
        tok_emb = self.token_embedding_table(idx)

        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)

        x = tok_emb + pos_emb

        # Transformer
        x = self.blocks(x)
        x = self.ln_f(x)

        # Logits for each possible next character
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
        """Generate new characters autoregressively."""

        for _ in range(max_new_tokens):

            # Keep only the most recent context
            idx_cond = idx[:, -block_size:]

            # Predict next character
            logits, _ = self(idx_cond)

            # Use the last time step
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next character
            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            # Append it to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ============================================================
# 12. INITIALIZE MODEL AND OPTIMIZER
# ============================================================

model = GPTLanguageModel().to(device)

num_params = sum(
    p.numel() for p in model.parameters()
)

print(f"\nModel parameters: {num_params:,}")
print(f"Model size: {num_params / 1e6:.3f} million parameters")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# 13. TRAIN THE MODEL
# ============================================================

print("\nStarting training...\n")

for iteration in range(max_iters):

    # Evaluate periodically
    if iteration % eval_interval == 0 or iteration == max_iters - 1:
        losses = estimate_loss()

        print(
            f"Step {iteration:4d} | "
            f"Train loss: {losses['train']:.4f} | "
            f"Validation loss: {losses['val']:.4f}"
        )

    # Get a training batch
    xb, yb = get_batch("train")

    # Forward pass
    logits, loss = model(xb, yb)

    # Backpropagation
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


print("\nTraining completed.")


# ============================================================
# 14. SAVE THE MODEL AND TOKENIZER
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

torch.save(checkpoint, "conversation_gpt.pt")

print("Checkpoint saved to conversation_gpt.pt")


# ============================================================
# 15. GENERATE CONVERSATION TEXT
# ============================================================

model.eval()

prompt = "Human 1:"

# Make sure every prompt character exists in the vocabulary
unknown_chars = [ch for ch in prompt if ch not in stoi]

if unknown_chars:
    raise ValueError(
        f"Prompt contains unknown characters: {unknown_chars}"
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
print("GENERATED CONVERSATION")
print("=" * 60)
print(generated_text)


# ============================================================
# 16. SAVE GENERATED TEXT TO A FILE
# ============================================================

with open("generated_conversation.txt", "w", encoding="utf-8") as f:
    f.write(generated_text)

print("\nGenerated text saved to generated_conversation.txt")
