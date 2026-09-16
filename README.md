# RSI experiments

Proyecto autocontenido para ejecutar autoresearch sobre el modelo inicial de
`v4.8_low_size_token.ipynb`.

Incluye el código del modelo LoopedGPT, las utilidades de entrenamiento, el
tokenizador SentencePiece FineWeb de 16.384 tokens, el script de preparación,
el entrenamiento y el programa de investigación.

## Preparación

1. Añade `data/smollm_local/train.txt` y
   `data/smollm_local/validation.txt` según las instrucciones de ese
   directorio.
2. Instala las dependencias: `uv sync`.
3. Comprueba los recursos: `uv run prepare.py`.
4. Ejecuta el baseline: `uv run train.py`.

Cada ejecución registra configuración, pérdidas y la métrica final
`test_loss` en Weights & Biases, dentro del proyecto `gpt2-50M`.

Consulta `program.md` para el protocolo de experimentación autónoma.

The autonomous search may modify `train.py` and the isolated `rsi_architecture/` candidate package while data, tokenizer, evaluation, and shared `src/llm_mini_lab/` code remain fixed.

## System documentation

The end-to-end authority model and operating procedures are documented in:

- [`docs/adr/ADR-0001-governed-autoresearch-evidence-boundaries.md`](docs/adr/ADR-0001-governed-autoresearch-evidence-boundaries.md)
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md)
- [`docs/README.md`](docs/README.md)

These documents distinguish candidate mutation, fixed measurement, W&B/ARIA advisory evidence, passive dynamic observation, visualization, and the separately reviewed local orchestration candidate.
