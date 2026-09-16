"""Preparación fija para autoresearch con los datos locales de 1M tokens.

Usa ``data/smollm_local/train.txt``, ``validation.txt`` y el tokenizador
``tokenizers/fineweb_16384_bpe.model`` del notebook v4.8_low_size_token.
"""

from pathlib import Path

import sentencepiece as spm
import torch

# No modificar durante la búsqueda autónoma.
MAX_SEQ_LEN = 128
TIME_BUDGET = 300
TRAIN_TOKENS = 1_000_000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "smollm_local"
TRAIN_PATH = DATA_DIR / "train.txt"
VAL_PATH = DATA_DIR / "validation.txt"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizers" / "fineweb_16384_bpe.model"


class Tokenizer:
    """Adaptador de SentencePiece compatible con el train.py original."""

    def __init__(self, processor):
        self.processor = processor
        self.bos_token_id = processor.bos_id()
        if self.bos_token_id < 0:
            self.bos_token_id = processor.eos_id()
        if self.bos_token_id < 0:
            self.bos_token_id = 0

    @classmethod
    def from_directory(cls, tokenizer_dir=None):
        del tokenizer_dir
        if not TOKENIZER_PATH.is_file():
            raise FileNotFoundError(f"No se encontró el tokenizador de 16K: {TOKENIZER_PATH}")
        return cls(spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH)))

    def get_vocab_size(self):
        return self.processor.vocab_size()

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        del num_threads
        if isinstance(text, str):
            ids = self.processor.encode(text, out_type=int)
            return ([prepend] if prepend is not None else []) + ids
        if isinstance(text, list):
            encoded = [self.processor.encode(item, out_type=int) for item in text]
            return ([[prepend] + ids for ids in encoded] if prepend is not None else encoded)
        raise TypeError(f"Tipo de texto no compatible: {type(text)}")

    def decode(self, ids):
        return self.processor.decode(ids)


def _read_documents(path, tokenizer, token_limit=None):
    """Tokeniza líneas locales y, para train, las limita a un millón de tokens."""
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el conjunto de datos: {path}")
    documents, tokens_seen = [], 0
    separator = tokenizer.get_bos_token_id()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ids = [separator] + tokenizer.encode(line)
        if token_limit is not None:
            remaining = token_limit - tokens_seen
            if remaining <= 0:
                break
            ids = ids[:remaining]
        if len(ids) > 1:
            documents.append(ids)
            tokens_seen += len(ids)
    if not documents:
        raise ValueError(f"{path} no contiene texto tokenizable")
    return documents, tokens_seen


def _document_stream(split, tokenizer):
    path = TRAIN_PATH if split == "train" else VAL_PATH
    limit = TRAIN_TOKENS if split == "train" else None
    documents, _ = _read_documents(path, tokenizer, limit)
    epoch = 1
    while True:
        for document in documents:
            yield document, epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """Lotes CUDA con empaquetado secuencial de los documentos locales."""
    del buffer_size
    if split not in {"train", "val"}:
        raise ValueError("split debe ser 'train' o 'val'")
    if not torch.cuda.is_available():
        raise RuntimeError("El train.py original de autoresearch requiere una GPU CUDA.")
    stream = _document_stream(split, tokenizer)
    current, epoch = next(stream)
    offset = 0
    while True:
        rows = []
        for _ in range(B):
            row = []
            while len(row) < T + 1:
                take = min(T + 1 - len(row), len(current) - offset)
                row.extend(current[offset:offset + take])
                offset += take
                if offset == len(current):
                    current, epoch = next(stream)
                    offset = 0
            rows.append(row)
        batch = torch.tensor(rows, dtype=torch.long, pin_memory=True).to("cuda", non_blocking=True)
        yield batch[:, :-1], batch[:, 1:], epoch


if __name__ == "__main__":
    tokenizer = Tokenizer.from_directory()
    train_docs, train_tokens = _read_documents(TRAIN_PATH, tokenizer, TRAIN_TOKENS)
    val_docs, val_tokens = _read_documents(VAL_PATH, tokenizer)
    print(f"Tokenizador: {TOKENIZER_PATH.name} ({tokenizer.get_vocab_size():,} tokens)")
    print(f"Entrenamiento: {train_tokens:,}/{TRAIN_TOKENS:,} tokens en {len(train_docs):,} documentos")
    print(f"Validación: {val_tokens:,} tokens en {len(val_docs):,} documentos")
    if train_tokens < TRAIN_TOKENS:
        print("Aviso: train.txt tiene menos de 1.000.000 de tokens; se usará completo.")
    print("Listo para ejecutar train.py.")
