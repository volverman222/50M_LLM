#!/usr/bin/env python3
"""Preentrena el Looped GPT de 50M sobre 1B de tokens de SmolLM Corpus.

El corpus se lee en streaming desde Hugging Face y, por defecto, utiliza el
subconjunto educativo ``cosmopedia-v2``.
"""

from train_pretrain_1b import main


if __name__ == "__main__":
    main(dataset="smollm")
