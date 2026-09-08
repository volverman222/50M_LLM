# 50M LLM — Pretraining Base

This repository contains the full pretraining pipeline migrated from
`My_LLM_project`, ready to be adapted to a model with at most 50M trainable
parameters.

## Included

- `src/llm_mini_lab/`: GPT implementation, causal attention, and reusable
  training/data-loading utilities.
- `notebooks/pretraining/full/`: dataset extraction, W&B data logging, and two
  external-CUDA full-pretraining notebooks.
- `data/instruction_data.json`: small companion training data from the source
  project.
- `checkpoints/`: destination for local checkpoints (weights are not included).

No pretrained weights, checkpoints, W&B run artifacts, or credentials were
migrated.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[training]"
```

## Run

Use `notebooks/Test_pretrain.ipynb` for a short smoke test of the 50M model.

For an optimized CUDA run with gradient accumulation, mixed precision,
checkpoint recovery, Weights & Biases, and optional Hugging Face snapshots,
use `notebooks/pretraining/full/pretrain_external_gpu_v2.ipynb`.

## Required adaptation before the challenge run

The inherited `GPT_CONFIG_124M` configuration is approximately 124M parameters
when token embedding/output weights are tied. It **exceeds the 50M parameter
limit**. Reduce the configuration in `src/llm_mini_lab/pretraining.py` (and
the matching notebook config) before training the submitted model, then verify
the printed parameter count.
