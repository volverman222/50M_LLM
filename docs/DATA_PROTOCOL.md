# Devpost Hackathon — 50M_LLM
## Explicit data protocol and project alignment

**State: local owner-review candidate.** Nothing in these profiles is authorization to start a new training run or publish changes.

### Why the new path is opt-in

The original training functions and their defaults remain in place. Without `--data-protocol`, the trainer uses its existing data loader. With the flag, it validates a separate JSON contract before creating a model or opening a dataset. This avoids changing the meaning of the team's prior runs while making new experiments explicit and comparable.

The reviewed upstream is `c6bdcc102af7513871c6cc09f2a0c3b1c02cb9df`. It contains GQA/tied-weight support and synthetic GQA comparison notebooks. Synthetic notebook timing runs are not corpus-quality evaluations. The preserved looped baseline is a separate model configuration, not silently converted to the newer GQA notebook architecture.

### What can be configured

| Boundary | Explicit settings |
|---|---|
| Source | Hugging Face repository, subset/config, immutable revision SHA, native split, text column; or local JSONL files and SHA256 identities |
| Tokenizer | GPT-2 or SP16K; SP16K file hash must match the configured tokenizer |
| Sequence | Token context length, packed stream versus per-document windows, EOS insertion, explicit incomplete-window discard |
| Batch | Microbatch size and whether to drop an incomplete batch; accumulation remains a training setting |
| Membership | Native train/validation/test splits, stable content-hash partition, or explicit legacy modulo mode |
| Traversal | Independent data-order seed, buffered shuffle size and whether to reseed training order per epoch |
| Limits | Optional positive document-count limit per role; target training tokens remain separate |
| Execution sequence | Success-dependent job ordering through RunSpec `depends_on`; not automatic curriculum continuation |

The current runtime deliberately accepts only `num_workers: 0`. Unsupported sharding/resume modes fail rather than quietly changing split membership or skipping a cursor mismatch. Packed and per-document windows both produce `x[t:t+T]` and its one-token-shifted target, requiring `T+1` source tokens. The original final-microbatch token rounding/accumulation behavior remains unchanged.

### Three distinct split modes

**`upstream_modulo`** preserves the original order-dependent scheme: shuffle the source using the fixed order seed, ignore empty text, then assign by nonempty-document index modulo N. Per-epoch reseeding is rejected because it would change membership. The separate split-seed field is not used in this strategy; the order seed determines the permutation. This compatibility profile is not a new leakage audit.

**`stable_hash`** assigns exact UTF-8 document content using `SHA256(str(split_seed) + NUL + text)` before changing training order. The first eight digest bytes, big-endian, determine the bucket modulo N. Validation/test bucket lists must be disjoint, in range and leave training buckets. Exact duplicate text receives the same role. This is not semantic or near-duplicate decontamination, and bucket proportions are approximate, not exact document quotas. Validation/test order stays fixed; training order can change independently between epochs.

**`source_splits`** selects named native splits and does not repartition them. Train and validation names are mandatory and distinct; test may be explicitly null. Local role files must have different paths. File names or hashes do not establish corpus disjointness: content overlap still needs a separate check.

### Verified source presets

`configs/data_protocols/source_catalog.json` records the metadata check and repository revisions. No corpus was downloaded for the audit.

- `smollm_cosmopedia_legacy_128.json`: the original SmolLM Cosmopedia v2 source with fixed order/modulo behavior, 128-token windows; no invented native validation split.
- `smollm_cosmopedia_hash_128.json`: prospective independent split/order example. Changing from the legacy split is a new experimental protocol, not a replay of old baseline validation.
- `fineweb_edu_hash_128.json`: `codelion/fineweb-edu-1B`, `default` configuration, pinned source, prospective hash split.
- `wikitext103_split_reference.json`: `Salesforce/wikitext`, specifically `wikitext-103-raw-v1`, with its real train/validation/test names. This is a **source/split reference**, not an official scoring recipe or permission to train on benchmark data.

All four have been checked against the protocol schema/runtime structure. A source path/subset/revision is distinct from a completed data-content verification. Run admission preserves the recipe and the trainer records the actual protocol receipt. Use resolved local-file paths for machine-specific datasets; do not copy somebody else's workstation path into a portable recipe.

### Direct training use

```bash
python scripts/train_pretrain_smollm_1b.py \
  --data-protocol configs/data_protocols/smollm_cosmopedia_legacy_128.json \
  --context-length 128 --micro-batch-size 16 --num-workers 0 \
  --tokenizer sp16384 --tokenizer-model tokenizers/fineweb_16384_bpe.model \
  --emb-dim 1024 --n-heads 8 --n-kv-heads 8 --n-unique-layers 3 --num-loops 2 \
  --ff-hidden-dim 1376 --positional-encoding rope --seed 123 \
  --target-tokens 100000000 --hf-upload-every 0 --hourly-cost 0 --dry-run
```

This command only validates the prospective configuration/model; it does not load data. Remove `--dry-run` only after reviewing a concrete experiment. The orchestrator's `prepare-ml` compiles these settings into a sealed run specification, with separate evidence paths and resource requirements.

### Official evaluation remains a distinct gate

The published track specifies HellaSwag, ARC-Easy, PIQA and WinoGrande via `lm-evaluation-harness`, plus held-out WikiText-103 perplexity. The repository's small custom benchmark helper and local Cosmopedia validation are useful diagnostics, but they are not replacements for that prescribed evaluation.

The current upstream harness task named `wikitext` selects **WikiText-2**, not WikiText-103. A generic `--tasks wikitext` invocation must not be relabelled as the required 103 result. The exact 103 variant, held-out slice, preprocessing, scoring denominator, window/stride and harness revision need to be pinned and reviewed. No numeric results are fabricated for that open gate; see `configs/evaluation/official_eval_plan.json`.

### Evidence and interpretation

Schema: `schemas/data_protocol.schema.json`; executable validation: `llm_mini_lab.data_protocol.validate`. Runtime validation adds cross-field bucket/path/hash and CLI checks beyond JSON Schema. The recorded protocol is cached for the process, so subsequent stream epochs do not reread an altered external configuration.

Source/config hashes preserve identity, not independent trust or semantic decontamination. Defensive checks screen possible failure modes; they do not assert that the team's architecture exhibits them.

Sources: repository files at the reviewed Git SHA; [official track rules](https://gibc-v2.devpost.com/rules); [Hugging Face dataset revision loading](https://huggingface.co/docs/datasets/loading); [harness WikiText task definition](https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/wikitext/wikitext.yaml).

### Queue-to-execution provenance guard
`prepare-ml` records the exact local source-file hashes and Git revision, including uncommitted Python changes. The executor rejects source drift before starting a process. The opt-in trainer records a shape-framed, ordered little-endian int64 input/target digest and counts after successful backward passes in `training_input_identity.json`; this is not a replay archive or semantic-decontamination claim. Browser and CLI expose consumed-input and validation identities separately from the requested data recipe. Old frozen baseline receipts are not rewritten.
