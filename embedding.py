from typing import List, Optional, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModel
import faiss

faiss.omp_set_num_threads(1)
import torch

torch.set_num_threads(1)

EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def get_embeddings(
    texts: List[str], tokenizer, model, device, batch_size: int = 4
) -> np.ndarray:
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            out = model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            last_hidden = out.last_hidden_state
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1)
                summed = (last_hidden * mask).sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1)
                emb = summed / lengths
            else:
                emb = last_hidden.mean(dim=1)
        emb = emb.cpu().numpy()
        all_emb.append(emb)
    emb_matrix = np.vstack(all_emb).astype("float32")
    faiss.normalize_L2(emb_matrix)
    return emb_matrix


def get_embeddings(
    texts: List[str],
    model_name: str = EMBEDDING_MODEL_NAME,
    device: Optional[str] = None,
    batch_size: int = 32,
) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype="auto")
    model.to(device)
    model.eval()

    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            out = model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            last_hidden = out.last_hidden_state
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1)
                summed = (last_hidden * mask).sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1)
                emb = summed / lengths
            else:
                emb = last_hidden.mean(dim=1)
        emb = emb.cpu().float().numpy()
        all_emb.append(emb)
    emb_matrix = np.vstack(all_emb).astype("float32")
    faiss.normalize_L2(emb_matrix)
    return emb_matrix


def build_faiss_index(phrases: List[str]) -> faiss.Index:
    """
    Returns a flat FAISS index.
    """
    embeddings = get_embeddings(phrases)
    d = embeddings.shape[1]

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIDMap(quantizer)
    custom_ids = np.arange(0, len(embeddings), dtype=np.int64)

    if not index.is_trained:
        index.train(embeddings)

    index.add_with_ids(embeddings, custom_ids)
    return index


def save_faiss_index(index: faiss.Index, file_path: str) -> None:
    """
    Save a FAISS index to `file_path`.
    """
    try:
        faiss.write_index(index, file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to save FAISS index to {file_path}: {e}") from e


def load_faiss_index(file_path: str) -> faiss.Index:
    """
    Load a FAISS index from `file_path` and return it.
    """
    try:
        index = faiss.read_index(file_path)
        return index
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index from {file_path}: {e}") from e


def search_faiss_index(
    index: faiss.Index, query_vector: np.ndarray, k: int = 5
) -> List[Tuple[int, float]]:
    """
    Search `index` for the top-`k` nearest vectors to `query_vector`.

    Returns a list of tuples `(vector_id, similarity_score)`. Similarity is the
    inner-product score (assumes vectors are L2-normalized for cosine similarity).
    """
    q = np.asarray(query_vector, dtype="float32")
    if q.ndim == 1:
        q = q.reshape(1, -1)
    elif q.ndim != 2 or q.shape[0] != 1:
        raise ValueError(
            "query_vector must be a 1-D vector or a 2-D array with shape (1, d)"
        )

    # normalize for cosine similarity
    faiss.normalize_L2(q)

    # clamp k to available entries
    ntotal = 0
    try:
        ntotal = index.ntotal
    except Exception:
        ntotal = 0
    if ntotal == 0:
        return []
    k = min(k, max(1, ntotal))

    # attempt search; if index is GPU-backed and fails, convert to CPU and retry
    D, I = index.search(q, k=k)
    results: List[Tuple[int, float]] = []
    for idx, score in zip(I[0].tolist(), D[0].tolist()):
        if idx < 0:
            continue
        results.append((int(idx), float(score)))

    return results
