"""Descarga una porcion de SmolLM Corpus y crea un split local 90/10."""

import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


DATASET = "HuggingFaceTB/smollm-corpus"
CONFIG = "cosmopedia-v2"
NUM_DOCUMENTS = 1_000
PAGE_SIZE = 100
OUTPUT_DIR = Path("data/smollm_local")
MAX_RETRIES = 5


def download_page(offset):
    query = urlencode({
        "dataset": DATASET,
        "config": CONFIG,
        "split": "train",
        "offset": offset,
        "length": PAGE_SIZE,
    })
    url = f"https://datasets-server.huggingface.co/rows?{query}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urlopen(url, timeout=60) as response:
                return json.load(response).get("rows", [])
        except (HTTPError, URLError, TimeoutError):
            if attempt == MAX_RETRIES:
                raise
            time.sleep(attempt * 5)


def extract_text(item):
    row = item.get("row", item)
    text = row.get("text", "")
    return text if isinstance(text, str) else ""


def write_document(output_file, text):
    output_file.write(text.strip())
    output_file.write("\n<|endoftext|>\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train.txt"
    validation_path = OUTPUT_DIR / "validation.txt"

    downloaded = 0
    train_count = 0
    validation_count = 0
    offset = 0

    with train_path.open("w", encoding="utf-8") as train_file, \
            validation_path.open("w", encoding="utf-8") as validation_file:
        while downloaded < NUM_DOCUMENTS:
            rows = download_page(offset)
            if not rows:
                break

            for item in rows:
                text = extract_text(item)
                if not text.strip():
                    continue

                # Como val_mod=10: cada decimo documento es validacion.
                if downloaded % 10 == 0:
                    write_document(validation_file, text)
                    validation_count += 1
                else:
                    write_document(train_file, text)
                    train_count += 1

                downloaded += 1
                if downloaded >= NUM_DOCUMENTS:
                    break

            offset += len(rows)
            print(f"Descargados: {downloaded}/{NUM_DOCUMENTS} documentos")

    if downloaded < NUM_DOCUMENTS:
        raise RuntimeError(
            f"El servidor solo proporciono {downloaded} documentos validos"
        )

    print(f"Train: {train_count} documentos -> {train_path.resolve()}")
    print(
        f"Validacion: {validation_count} documentos -> "
        f"{validation_path.resolve()}"
    )


if __name__ == "__main__":
    main()
