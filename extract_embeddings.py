# extract_embeddings.py

import gc
import json
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from config import TEXT_DIR, EMB_DIR, TEXT_MODELS, MAX_LENGTH, BATCH_SIZE, SPLITS


# =====================================================
# POOLING
# =====================================================

def mean_pooling(last_hidden_state, attention_mask):
    """
    Mean pooling usando attention_mask para ignorar padding.
    Devuelve un embedding por paciente.
    """

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask

    summed = masked_embeddings.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)

    return summed / counts


# =====================================================
# EXTRACCIÓN DE EMBEDDINGS
# =====================================================

@torch.no_grad()
def extract_embeddings_for_split(model_name_key, hf_model_name, split_name, device):
    print("\n" + "=" * 80)
    print(f"Modelo: {model_name_key} | Split: {split_name}")
    print("=" * 80)

    text_path = TEXT_DIR / f"{split_name}_texts.csv"

    if not text_path.exists():
        raise FileNotFoundError(
            f"No existe {text_path}. "
            "Primero ejecuta build_patient_text.py con la estrategia de split actual."
        )

    df = pd.read_csv(text_path)

    if "text" not in df.columns:
        raise ValueError(f"No encontré columna 'text' en {text_path}")

    if "label" not in df.columns:
        raise ValueError(f"No encontré columna 'label' en {text_path}")

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).values

    print("Número de textos:", len(texts))
    print("Distribución labels:", pd.Series(labels).value_counts().to_dict())

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModel.from_pretrained(hf_model_name)

    model.to(device)
    model.eval()

    max_length = MAX_LENGTH[model_name_key]
    batch_size = BATCH_SIZE[model_name_key]

    print("max_length:", max_length)
    print("batch_size:", batch_size)

    all_embeddings = []

    for i in tqdm(
        range(0, len(texts), batch_size),
        desc=f"{model_name_key}-{split_name}",
    ):
        batch_texts = texts[i: i + batch_size]

        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        tokens = {
            k: v.to(device)
            for k, v in tokens.items()
        }

        outputs = model(**tokens)

        emb = mean_pooling(
            outputs.last_hidden_state,
            tokens["attention_mask"],
        )

        emb = emb.detach().cpu().numpy()
        all_embeddings.append(emb)

        del tokens, outputs, emb

        if device == "cuda":
            torch.cuda.empty_cache()

    embeddings = np.vstack(all_embeddings)

    emb_out = EMB_DIR / f"{model_name_key}_{split_name}_embeddings.npy"
    label_out = EMB_DIR / f"{model_name_key}_{split_name}_labels.npy"

    np.save(emb_out, embeddings)
    np.save(label_out, labels)

    metadata = {
        "model_name_key": model_name_key,
        "hf_model_name": hf_model_name,
        "split": split_name,
        "text_path": str(text_path),
        "embedding_path": str(emb_out),
        "labels_path": str(label_out),
        "n_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "device": device,
        "label_distribution": {
            str(k): int(v)
            for k, v in pd.Series(labels).value_counts().to_dict().items()
        },
    }

    metadata_out = EMB_DIR / f"{model_name_key}_{split_name}_metadata.json"

    with open(metadata_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nGuardado:")
    print("Embeddings:", emb_out)
    print("Labels    :", label_out)
    print("Metadata  :", metadata_out)
    print("Embeddings shape:", embeddings.shape)

    del model, tokenizer, all_embeddings, embeddings
    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()


# =====================================================
# MAIN
# =====================================================

def main():
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if len(TEXT_MODELS) == 0:
        raise ValueError("TEXT_MODELS está vacío en config.py.")

    for model_name_key, hf_model_name in TEXT_MODELS.items():
        for split_name in SPLITS:
            extract_embeddings_for_split(
                model_name_key=model_name_key,
                hf_model_name=hf_model_name,
                split_name=split_name,
                device=device,
            )

    print("\nExtracción de embeddings terminada.")


if __name__ == "__main__":
    main()