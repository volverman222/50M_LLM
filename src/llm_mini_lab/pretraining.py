"""
GPT pretraining (section 5.4 of the book "Build a Large Language Model
From Scratch"): trains GPT-2 small (124M) from scratch on the streaming
codelion/fineweb-edu-1B dataset from Hugging Face. Requires: pip install datasets

Only the essentials: dataset + dataloader, loss function, training loop,
and sample generation. No TensorFlow required.

Import the reusable functions from a notebook to configure and run training.
"""

import math
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

import tiktoken

# ---------------------------------------------------------------------------
# Model configuration (same as in chapter 5, section 5.4)
# ---------------------------------------------------------------------------
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # GPT-2 vocabulary
    "context_length": 256,    # context window from chapter 5
    "emb_dim": 768,           # embedding dimension
    "n_heads": 12,            # attention heads
    "n_layers": 12,           # transformer blocks
    "drop_rate": 0.0,         # no dropout during pretraining
    "qkv_bias": False         # no bias in Q/K/V (book config)
}

GPT_CONFIG_50M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 512,
    "n_heads": 8,
    "n_layers": 7,
    "drop_rate": 0.0,
    "qkv_bias": False,
}

LOOPED_GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 512,
    "n_heads": 8,

    # Arquitectura recurrente
    "n_unique_layers": 3,
    "num_loops": 2,

    "drop_rate": 0.0,
    "qkv_bias": False,
    "ff_activation": "swiglu",
    "ff_hidden_dim": 1376,
}
# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
class LocalTextDataset(Dataset):
    """Dataset of contiguous, non-overlapping token blocks from a text file."""

    def __init__(self, file_path, max_length):
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

        text = file_path.read_text(encoding="utf-8")
        tokenizer = tiktoken.get_encoding("gpt2")
        self.tokens = tokenizer.encode(
            text, allowed_special={"<|endoftext|>"}
        )
        self.max_length = max_length

        if len(self.tokens) <= max_length:
            raise ValueError(f"{file_path} no contiene suficientes tokens")

    def __len__(self):
        return (len(self.tokens) - 1) // self.max_length

    def __getitem__(self, index):
        start = index * self.max_length
        end = start + self.max_length
        inputs = torch.tensor(self.tokens[start:end], dtype=torch.long)
        targets = torch.tensor(
            self.tokens[start + 1:end + 1], dtype=torch.long
        )
        return inputs, targets


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # shape: (1, n_tokens)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # Greedy (argmax) generation used to show samples during training
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# ---------------------------------------------------------------------------
# Loss, evaluation and training (section 5.4)
# ---------------------------------------------------------------------------
def init_xavier(module):
    """Initialize linear and embedding weights with Xavier uniform values."""
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.xavier_uniform_(module.weight)


def evaluate(model, data_loader, max_batches, device=None):
    """Return mean cross-entropy loss while restoring the model's mode."""
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration as exc:
            raise ValueError(
                "device is required when the model has no parameters"
            ) from exc

    was_training = model.training
    model.eval()
    total_loss = 0.0
    batches_evaluated = 0

    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.flatten(0, 1), targets.flatten()
            )
            total_loss += loss.item()
            batches_evaluated += 1

    model.train(was_training)

    if batches_evaluated == 0:
        raise RuntimeError("El dataloader de validación no produjo batches")

    return total_loss / batches_evaluated


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(
        logits.flatten(0, 1),  # (batch*seq, vocab)
        target_batch.flatten()  # (batch*seq,)
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    try:
        n_total = len(data_loader)
    except TypeError:
        n_total = None  # streaming dataloader (IterableDataset): no len()

    if n_total == 0:
        return float("nan")
    if num_batches is None:
        if n_total is None:
            raise ValueError(
                "num_batches is required for streaming dataloaders")
        num_batches = n_total
    elif n_total is not None:
        num_batches = min(num_batches, n_total)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def cosine_lr_multiplier(step, warmup_steps, schedule_steps,
                         min_lr_ratio=0.1):
    """Linear warmup followed by cosine decay to ``min_lr_ratio``."""
    if warmup_steps <= 0 or schedule_steps <= warmup_steps:
        raise ValueError("schedule_steps must be greater than warmup_steps > 0")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (schedule_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine


def evaluate_loader_mixed_precision(loader, max_batches, *, model, device,
                                    autocast_dtype=torch.float16):
    """Evaluate a loader using autocast, restoring the model's train state."""
    was_training = model.training
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
            losses.append(loss.float().item())
    model.train(was_training)
    if not losses:
        raise ValueError("The evaluation loader produced no batches")
    return sum(losses) / len(losses)


def save_training_checkpoint(update_step, microbatch_idx, tokens_seen,
                             history, suffix, *, checkpoint_dir,
                             filename_prefix, model, optimizer, scheduler,
                             scaler, config, wandb_run_id=None):
    """Save all state required to resume an external-GPU training run."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{filename_prefix}-{suffix}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "update_step": update_step,
        "microbatch_idx": microbatch_idx,
        "tokens_seen": tokens_seen,
        "history": history,
        "config": config,
        "wandb_run_id": wandb_run_id,
    }, path)
    return path


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer,
                       eval_train_loader=None, eval_val_loader=None):
    """Training loop. By default it evaluates on train_loader/val_loader;
    with streaming datasets pass fixed loaders in eval_train_loader/eval_val_loader."""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    eval_train = eval_train_loader or train_loader
    eval_val = eval_val_loader or val_loader

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, eval_train, eval_val, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


# ---------------------------------------------------------------------------
# FineWeb-Edu (codelion/fineweb-edu-1B): large HuggingFace dataset, streamed
# (not downloaded entirely). Same pattern as the book's dataloader
# (ch05/10_llm-training-speed): non-overlapping windows + <|endoftext|> token
# between documents. Requires: pip install datasets
# ---------------------------------------------------------------------------
class TokenBlockIterableDataset(IterableDataset):
    """Streaming: tokenizes FineWeb-Edu documents and yields max_length-token
    windows. Each document ends with <|endoftext|>. Documents are split into
    train/val according to val_mod (every val_mod-th document goes to
    validation)."""

    def __init__(self, stream, encoder, max_length=1024, add_eot=True,
                 val_mod=100, val_split=False, max_docs=None,
                 max_tokens=None, show_progress=False,
                 progress_description="Dataset: descargando/tokenizando"):
        self.stream = stream
        self.max_length = max_length
        self.add_eot = add_eot
        self.enc = encoder
        self.val_mod = val_mod
        self.val_split = val_split
        self.max_docs = max_docs
        self.max_tokens = max_tokens
        self.show_progress = show_progress
        self.progress_description = progress_description
        self.EOT = encoder.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    def __iter__(self):
        it = self.stream
        wi = get_worker_info()
        if wi is not None and wi.num_workers > 1:
            it = self.stream.shard(
                num_shards=wi.num_workers, index=wi.id, contiguous=True)

        progress = None
        if self.show_progress and self.max_tokens is not None:
            try:
                from tqdm.auto import tqdm
                progress = tqdm(
                    total=self.max_tokens, unit="tok", unit_scale=True,
                    desc="Dataset: conectando al stream")
            except ImportError:
                pass

        buf = []
        seen, kept, tokenized = 0, 0, 0
        try:
            for ex in it:
                text = ex.get("text", "") if ex else ""
                if not isinstance(text, str) or not text.strip():
                    continue
                if (seen % self.val_mod == 0) != self.val_split:
                    seen += 1
                    continue
                seen += 1

                toks = self.enc.encode(text)
                if self.add_eot:
                    toks.append(self.EOT)
                if self.max_tokens is not None:
                    remaining = self.max_tokens - tokenized
                    if remaining <= 0:
                        break
                    toks = toks[:remaining]
                tokenized += len(toks)
                if progress is not None:
                    if progress.n == 0:
                        progress.set_description(self.progress_description)
                    progress.update(len(toks))
                buf.extend(toks)

                while len(buf) >= self.max_length + 1:
                    x = torch.tensor(buf[:self.max_length], dtype=torch.long)
                    y = torch.tensor(buf[1:self.max_length + 1], dtype=torch.long)
                    yield x, y
                    del buf[:self.max_length]

                kept += 1
                if ((self.max_docs is not None and kept >= self.max_docs) or
                        (self.max_tokens is not None and
                         tokenized >= self.max_tokens)):
                    break
        finally:
            if progress is not None:
                progress.close()


class FixedPairsDataset(Dataset):
    """In-memory dataset (with len()) holding already tokenized (x, y) pairs."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class HFDatasetRowsAPI:
    """Small, restartable Hugging Face Dataset Viewer rows stream.

    This avoids opening large Parquet shards and is intended for smoke tests,
    not high-throughput pretraining.
    """

    def __init__(self, dataset, config, split="train", page_size=100,
                 start_offset=0, timeout=60):
        if not 1 <= page_size <= 100:
            raise ValueError("page_size must be between 1 and 100")
        self.dataset = dataset
        self.config = config
        self.split = split
        self.page_size = page_size
        self.start_offset = start_offset
        self.timeout = timeout

    def __iter__(self):
        offset = self.start_offset
        while True:
            query = urlencode({
                "dataset": self.dataset,
                "config": self.config,
                "split": self.split,
                "offset": offset,
                "length": self.page_size,
            })
            url = f"https://datasets-server.huggingface.co/rows?{query}"
            with urlopen(url, timeout=self.timeout) as response:
                payload = json.load(response)
            rows = payload.get("rows", [])
            if not rows:
                break
            for item in rows:
                yield item["row"]
            offset += len(rows)
            if len(rows) < self.page_size:
                break


def materialize_from_loader(loader, max_batches=32):
    """Freezes the first max_batches of a streaming loader into a Dataset."""
    pairs = []
    for i, (x, y) in enumerate(loader):
        pairs.extend(list(zip(x, y)))
        if (i + 1) >= max_batches:
            break
    return FixedPairsDataset(pairs)


def make_fixed_eval_loaders(train_loader, val_loader,
                            max_train_batches=8, max_val_batches=16):
    """Fixed loaders (with len()) for evaluating without disturbing the stream."""
    train_bs = getattr(train_loader, "batch_size", 1) or 1
    val_bs = getattr(val_loader, "batch_size", 1) or 1

    fixed_train_eval = materialize_from_loader(
        train_loader, max_batches=max_train_batches)
    fixed_val_eval = materialize_from_loader(
        val_loader, max_batches=max_val_batches)

    train_eval_loader = DataLoader(
        fixed_train_eval, batch_size=train_bs, shuffle=False, drop_last=True)
    val_eval_loader = DataLoader(
        fixed_val_eval, batch_size=val_bs, shuffle=False, drop_last=True)
    return train_eval_loader, val_eval_loader


def create_dataloader_fineweb(batch_size=2, max_length=1024, val_mod=100,
                              seed=123, max_docs=None, num_workers=0,
                              add_eot=True, shuffle_buffer=10000,
                              max_tokens=None, show_progress=False):
    """Streaming loaders of codelion/fineweb-edu-1B (~970k docs, ~1B tokens).

    Returns (train_loader, val_loader). To evaluate during training use
    make_fixed_eval_loaders(train_loader, val_loader).
    max_docs limits how many documents train consumes (None = the whole
    dataset, ~1B tokens). Each document ≈ 1-2k tokens. max_tokens limits
    the tokenized training stream precisely (apart from a final incomplete
    max_length block that cannot be yielded). show_progress displays token
    progress for the training stream when max_tokens is set.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "To use FineWeb-Edu you need to install 'datasets': "
            "pip install datasets") from e

    stream = load_dataset("codelion/fineweb-edu-1B",
                          split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)

    enc = tiktoken.get_encoding("gpt2")
    train_ds = TokenBlockIterableDataset(
        stream, enc, max_length=max_length, add_eot=add_eot,
        val_mod=val_mod, val_split=False, max_docs=max_docs,
        max_tokens=max_tokens, show_progress=show_progress)
    val_ds = TokenBlockIterableDataset(
        stream, enc, max_length=max_length, add_eot=add_eot,
        val_mod=val_mod, val_split=True, max_docs=None)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        drop_last=True, num_workers=num_workers)
    return train_loader, val_loader


def create_dataloader_smollm(batch_size=2, max_length=1024, val_mod=100,
                             seed=123, max_docs=None, num_workers=0,
                             add_eot=True, shuffle_buffer=100,
                             max_tokens=None, show_progress=False,
                             config_name="cosmopedia-v2",
                             use_rows_api=False, rows_page_size=100):
    """Stream a text subset from HuggingFaceTB/smollm-corpus.

    ``cosmopedia-v2`` is the default general-purpose educational subset.
    The token/document limits and train/validation split behave like the
    FineWeb loader, without downloading the complete corpus.
    """
    if use_rows_api:
        if num_workers != 0:
            raise ValueError("use_rows_api requires num_workers=0")
        stream = HFDatasetRowsAPI(
            "HuggingFaceTB/smollm-corpus", config_name,
            page_size=rows_page_size)
    else:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "To use SmolLM Corpus you need to install 'datasets': "
                "pip install datasets") from e
        stream = load_dataset(
            "HuggingFaceTB/smollm-corpus", config_name,
            split="train", streaming=True)
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)

    enc = tiktoken.get_encoding("gpt2")
    common = dict(
        stream=stream, encoder=enc, max_length=max_length, add_eot=add_eot,
        val_mod=val_mod)
    train_ds = TokenBlockIterableDataset(
        **common, val_split=False, max_docs=max_docs,
        max_tokens=max_tokens, show_progress=show_progress,
        progress_description="SmolLM: descargando/tokenizando")
    val_ds = TokenBlockIterableDataset(
        **common, val_split=True, max_docs=None)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        drop_last=True, num_workers=num_workers)
    return train_loader, val_loader
