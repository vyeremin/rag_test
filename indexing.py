import sys
import logging
from pathlib import Path
from typing import Dict, List

from embedding import build_faiss_index, save_faiss_index
from loader import load_texts_by_file, save_document_mapping


DATA_LOAD_PATH = "data/milestones/*.txt"
DOCUMENT_INDEX_FILE = "data/index/embedding_index.bin"
DOCUMENT_MAPPING_FILE = "data/index/documents.pickle"
SENTENCE_INDEX_FILE = "data/index/sentence_index.bin"
SENTENCE_MAPPING_FILE = "data/index/sentences.pickle"


def split_by_sentence(text: str) -> List[str]:
    """
    A simple sentence splitter that splits text into sentences based on punctuation.
    This is a naive implementation and may not handle all edge cases (e.g., abbreviations, quotes).
    """
    import re

    # Split on ., !, ?, followed by a space or end of string
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip() != ""
    ]


def create_sentences_with_context(text: str) -> List[str]:
    """
    Given a text split into sentences and return a list of sentences
    that include first sentence as prefix ("{sentence_0}: {sentence_i}").
    """
    sentences = split_by_sentence(text)
    if not sentences:
        return []
    # Naive approach to add some "thinking"/"context" to the sentences,
    # for demonstration purposes.
    prefix = sentences[0].split(",")[0]
    for i, sentence in enumerate(sentences):
        sentences[i] = sentence.replace("Their", "Childrens")
        sentences[i] = sentences[i].replace("They", "Children")
    if len(sentences) == 1:
        return sentences
    return ["%s: %s" % (prefix, sentence) for sentence in sentences[1:]]


def build_index_if_needed(data_load_path: str = DATA_LOAD_PATH) -> None:
    logger = logging.getLogger("query.build_index")
    if (
        not Path(DOCUMENT_INDEX_FILE).exists()
        or not Path(SENTENCE_INDEX_FILE).exists()
        or not Path(DOCUMENT_MAPPING_FILE).exists()
        or not Path(SENTENCE_MAPPING_FILE).exists()
    ):
        logger.debug("Building index for documents %s", data_load_path)
        texts = load_texts_by_file(data_load_path)
        sentences = []
        for key, lines in texts.items():
            header = lines[0]
            # Naive approach to add some "thinking"/"context" to the sentences, for demonstration purposes.
            if header.startswith("From") or header.startswith("Between") or header.startswith("During"):
                context = header.split(",")[0]
                line = ". ".join(lines)
                line = line.replace("Their", "Childrens")
                line = line.replace("They", "Children")
                passages = [passage for passage in line.split(".") if passage.strip() != ""]
                sentences.append(passages[0].replace(',', ':', 1).strip())
                sentences.extend("%s: %s" % (context, line.strip()) for line in passages[1:])
            else:
                # for "noise" files we use first line as context and last line as footer, and split the middle part into passages by empty lines
                context = header
                footer = lines[-1]
                passages = []
                for idx, line in enumerate(lines[1:-1]):
                    if line.strip() == "":
                        if len(passages) > 0:
                            sentences.append("%s: %s. %s" % (context, ". ".join(passages), footer))
                            passages = []
                        continue
                    passages.append(line.strip(' .'))
                if len(passages) > 0:
                    sentences.append("%s: %s. %s" % (context, ". ".join(passages), footer))
                texts[key] = [". ".join([line.strip(' .') for line in lines if line.strip() != ""])]
        documents = [line for lines in texts.values() for line in lines if line.strip() != ""]
        # log documents and sentences for debugging
        for idx, document in enumerate(documents):
            logger.debug("Document %i: %s", idx, document[:100])
        for idx, sentence in enumerate(sentences):
            logger.debug("Sentence %i: %s", idx, sentence[:100])

        faiss_document_index = build_faiss_index(documents)
        faiss_sentence_index = build_faiss_index(sentences)
        save_faiss_index(faiss_document_index, DOCUMENT_INDEX_FILE)
        save_faiss_index(faiss_sentence_index, SENTENCE_INDEX_FILE)
        save_document_mapping(documents, DOCUMENT_MAPPING_FILE)
        save_document_mapping(sentences, SENTENCE_MAPPING_FILE)


def main(argv=None):
    logger = logging.getLogger("indexing")
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        logger.error("Usage: python indexing.py <data_file_pattern>")
        return
    folder = argv[0]
    build_index_if_needed(folder)


if __name__ == "__main__":
    # default logging for standalone runs
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main()
