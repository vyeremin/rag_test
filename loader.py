import pickle
import sys
from pathlib import Path
from typing import Dict, List

def load_texts_by_file(path_pattern: str) -> Dict[str, List[str]]:
    """
    Load texts from files matching the given wildcard `path_pattern` (supports glob
    patterns, including recursive `**`). Returns a mapping from filename to a list
    of lines for each matched file.
    """
    import glob

    matches = glob.glob(path_pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"No files match pattern: {path_pattern}")

    texts: Dict[str, List[str]] = {}
    for file_path in matches:
        txt_path = Path(file_path)
        if not txt_path.is_file():
            continue
        try:
            content = txt_path.read_text(encoding="utf-8")
        except Exception:
            content = txt_path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        if not lines:
            continue
        texts[txt_path.name] = [line.strip() for line in lines]
    return texts

def save_document_mapping(documents: List[str], file_path: str) -> None:
    """
    Save a list of documents to `file_path` using pickle.
    """
    documents_id_map = {idx: lines for idx, lines in enumerate(documents)}
    with open(file_path, "wb") as f:
        pickle.dump(documents_id_map, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_document_mapping(file_path: str) -> Dict[int, str]:
    """
    Load a document mapping from `file_path` using pickle. Returns a mapping from
    document ID (int) to document text (str).
    """
    with open(file_path, "rb") as f:
        documents = pickle.load(f)
    return documents
