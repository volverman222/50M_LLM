# RSI experiments: research program

This directory applies the autoresearch loop to the initial model from
`notebooks/v4.8_low_size_token.ipynb`.

The goal is to reduce **`test_loss`**: the mean cross-entropy loss measured on
`data/smollm_local/validation.txt`. Lower is better. BPB is not used.

## Fixed resources

Do not modify these resources during experiments:

- `prepare.py`: local-data definitions and auxiliary checks.
- `data/smollm_local/train.txt`: training data.
- `data/smollm_local/validation.txt`: evaluation set.
- `tokenizers/fineweb_16384_bpe.model`: the 16,384-entry SentencePiece
  tokenizer.
- Code under `src/llm_mini_lab/` and the project dependencies.

Each run has a fixed budget: `MAX_TOKENS = 1_000_000`, which with batch size 2
and context length 128 equals 3,906 updates (999,936 tokens). Do not increase
this budget or use external data to improve the metric.

## Editable file

Only edit `train.py`. Prefer changes in the marked editable block:

- optimizer hyperparameters;
- batch size and context length, if they still work with the data;
- `LoopedGPTModel` configuration;
- architecture, while preserving the 50M-parameter limit;
- training-loop details that do not alter the budget or evaluation.

The baseline configuration replicates the notebook: `LoopedGPTModel`, RoPE,
three unique layers, eight heads, `emb_dim=1024`, AdamW, learning rate `3e-4`,
context length 128, and batch size 2.

Do not install packages, change the tokenizer or data files, add benchmarks to
the selection decision, or modify how `test_loss` is calculated.

## Setup

Before the first experiment:

1. Check that `train.txt`, `validation.txt`, and the tokenizer model exist.
2. Run `uv run prepare.py` from `rsi_exp/` to validate the resources.
3. Create `results.tsv` with the header specified below.
4. Run the unmodified `train.py` once. This result is the baseline.

Each run also creates a Weights & Biases run in the `gpt2-50M` project. It logs
the configuration, training loss, learning rate, tokens seen, `test/loss`, and
the final `test_loss` summary.

## Expected output

At the end, `train.py` must print one unambiguous line:

```
test_loss=0.000000
```

It may print intermediate losses, but only the final `test_loss` determines
whether an experiment is kept.

## Logging results

Record each attempt in `results.tsv` (TSV, not CSV) and do not include it in
commits:

```
commit	test_loss	status	description
```

Example:

```
commit	test_loss	status	description
a1b2c3d	2.843210	keep	baseline
b2c3d4e	2.801100	keep	increase learning rate
c3d4e5f	2.856900	discard	use four unique layers
d4e5f6g	0.000000	crash	batch size too large
```

## Experiment loop

1. Start from the commit with the best known `test_loss`.
2. Form one concrete hypothesis and make one small change in `train.py`.
3. Commit the change and run `uv run train.py > run.log 2>&1` from this
   directory.
4. Extract the metric with `rg '^test_loss=' run.log`.
5. If the process fails or does not produce that line, inspect the end of
   `run.log`, record a `crash`, and return to the last good commit.
6. Record the result. Keep the commit only if its `test_loss` is lower than the
   previous best; otherwise revert it.

Prefer simple, reproducible changes. A tiny improvement that introduces a lot
of complexity is not worth keeping. If an experiment changes memory use, note
the risk in its description.
