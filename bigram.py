
import torch
import torch.nn as nn
from torch.nn import functional as F
from urllib.request import urlopen

# ============================================================
# 1. HYPERPARAMETERS
# ============================================================

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 100

# Device selection
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(1337)

print("Using device:", device)


# ============================================================
# 2. LOAD YOUR CONVERSATION DATASET FROM GITHUB
# ============================================================

url = (
    "https://raw.githubusercontent.com/"
    "MandishaShivam/nanogpt-model/main/master/data/"
    "conversation-input.txt"
)

try:
    with urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8")
except Exception as e:
    raise RuntimeError(
        "Could not load the dataset from GitHub. "
        "Check your internet connection and repository URL."
    ) from e

print("Dataset loaded successfully.")
print("Total characters:", len(text))
print("\nDataset preview:\n")
print(text[:500])


# ============================================================
# 3. CREATE CHARACTER VOCABULARY AND TOKENIZER
# ============================================================

chars = sorted(list(set(text)))
vocab_size = len(chars)

# Character to integer
stoi = {ch: i for i, ch in enumerate(chars)}

# Integer to character
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    """Convert text to a list of character IDs."""
    return [stoi[c] for c in s]

def decode(ids):
    """Convert character IDs back into text."""
    return "".join(itos[int(i)] for i in ids)

print("\nVocabulary size:", vocab_size)


# ============================================================
# 4. ENCODE DATA AND CREATE TRAIN/VALIDATION SPLITS
# ============================================================

data = torch.tensor(encode(text), dtype=torch.long)

# First 90% for training, last 10% for validation
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

print("Training characters:", len(train_data))
print("Validation characters:", len(val_data))

if len(train_data) <= block_size or len(val_data) <= block_size:
    raise ValueError(
        "Dataset is too short for the selected block_size."
    )


# ============================================================
# 5. CREATE RANDOM TRAINING BATCHES
# ============================================================

def get_batch(split):
    """
    Generate input x and target y batches.
    Targets are shifted forward by one character.
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
# 7. BIGRAM LANGUAGE MODEL
# ============================================================

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # Each input character maps directly to logits
        # representing probabilities for the next character.
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            vocab_size
        )

    def forward(self, idx, targets=None):

        # idx shape: (B, T)
        # logits shape: (B, T, vocab_size)
        logits = self.token_embedding_table(idx)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Generate new characters one at a time.
        Each prediction depends only on the current character.
        """

        for _ in range(max_new_tokens):

            # Predict the next character
            logits, _ = self(idx)

            # Select predictions from the final time step
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next character
            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            # Append the sampled character
            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx


# ============================================================
# 8. INITIALIZE MODEL AND OPTIMIZER
# ============================================================

model = BigramLanguageModel(vocab_size).to(device)

num_params = sum(
    p.numel() for p in model.parameters()
)

print(f"\nModel parameters: {num_params:,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# 9. TRAIN THE MODEL
# ============================================================

print("\nStarting training...\n")

for iteration in range(max_iters):

    # Periodically evaluate training and validation loss
    if iteration % eval_interval == 0:
        losses = estimate_loss()

        print(
            f"Step {iteration:4d} | "
            f"Train loss: {losses['train']:.4f} | "
            f"Validation loss: {losses['val']:.4f}"
        )

    # Sample a batch
    xb, yb = get_batch("train")

    # Forward pass
    logits, loss = model(xb, yb)

    # Backpropagation and parameter update
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# Final loss evaluation
losses = estimate_loss()

print("\nTraining completed.")
print(f"Final train loss: {losses['train']:.4f}")
print(f"Final validation loss: {losses['val']:.4f}")


# ============================================================
# 10. SAVE MODEL AND VOCABULARY
# ============================================================

checkpoint = {
    "model_state_dict": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "vocab_size": vocab_size,
    "block_size": block_size,
}

torch.save(checkpoint, "conversation_bigram.pt")

print("\nModel saved to conversation_bigram.pt")


# ============================================================
# 11. GENERATE CONVERSATION TEXT
# ============================================================

model.eval()

# Start with a character that exists in the vocabulary
prompt = "H"

if any(ch not in stoi for ch in prompt):
    raise ValueError("The prompt contains an unknown character.")

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
print("GENERATED TEXT")
print("=" * 60)
print(generated_text)


# ============================================================
# 12. SAVE GENERATED TEXT
# ============================================================

with open(
    "generated_conversation.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(generated_text)

print("\nGenerated text saved to generated_conversation.txt")
