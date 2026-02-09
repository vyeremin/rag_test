import sys
import argparse
import logging
from typing import List
import numpy as np

from pathlib import Path

from analyzer import cleanup_query
from indexing import (
    DOCUMENT_INDEX_FILE,
    DOCUMENT_MAPPING_FILE,
    SENTENCE_INDEX_FILE,
    SENTENCE_MAPPING_FILE,
    build_index_if_needed,
)
from prompt import build_prompt
from embedding import load_faiss_index, get_embeddings, search_faiss_index
from loader import load_document_mapping
from inference import generate_response, generate_response_with_llm
from ranking import filter_results, rescore_results
from responser import generate_response_with_template


DOCUMENT_TOP_K = 3
DOCUMENT_SIMILARITY_THRESHOLD = 0.7

SENTENCE_TOP_K = 3
SENTENCE_SIMILARITY_THRESHOLD = 0.8

ANSWER_SIMILARITY_THRESHOLD = 0.7


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run query against FAISS index")
    parser.add_argument("query", nargs=1, help="Query string to search for")
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug/verbose output (sets log level to DEBUG)",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set log level (overrides -d)",
    )
    parser.add_argument(
        "--use-template",
        action="store_true",
        help="Use a template to generate response instead of LLM",
    )
    parser.add_argument(
        "--use-thinking", action="store_true", help="Use thinking feature for the model"
    )
    args = parser.parse_args(argv)

    query = args.query[0]
    use_template = args.use_template
    use_thinking = args.use_thinking
    # configure logging
    if args.log_level:
        level = getattr(logging, args.log_level)
    elif args.debug:
        level = logging.DEBUG
    else:
        level = logging.ERROR
    logging.getLogger("httpcore").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")
    logger = logging.getLogger("query")
    logger.debug("Received query: %s", query)

    # Validate and clean up the query
    if not query or query.strip() == "":
        logger.error("Query cannot be empty.")
        return
    if len(query) > 512:
        logger.warning(
            "Query is very long (%d characters). Consider shortening it.", len(query)
        )
    query = cleanup_query(query)
    if not query:
        logger.error("Invalid query.")
        return
    logger.debug("Used query: %s", query)

    build_index_if_needed()
    # load index and document mapping
    if use_template:
        logger.debug("Loading existing sentence index from %s...", SENTENCE_INDEX_FILE)
        faiss_index = load_faiss_index(SENTENCE_INDEX_FILE)
        document_mapping = load_document_mapping(SENTENCE_MAPPING_FILE)
        threshold = SENTENCE_SIMILARITY_THRESHOLD
        top_k = SENTENCE_TOP_K
    else:
        logger.debug("Loading existing document index from %s...", DOCUMENT_INDEX_FILE)
        faiss_index = load_faiss_index(DOCUMENT_INDEX_FILE)
        document_mapping = load_document_mapping(DOCUMENT_MAPPING_FILE)
        threshold = DOCUMENT_SIMILARITY_THRESHOLD
        top_k = DOCUMENT_TOP_K

    query_embedding = get_embeddings([query])
    results = search_faiss_index(faiss_index, query_embedding, k=top_k)

    # log concise search results
    logger.info("Top %i similar results for query", len(results))
    for idx, score in results:
        sentence = document_mapping.get(idx, "<missing>")
        logger.info("ID: %i (%.4f): %s", idx, score, sentence)

    # remove results that do not meet similarity threshold and re-score
    results = filter_results(results, threshold)
    results = rescore_results(results, document_mapping)
    if not results:
        print(
            "I'm sorry, but I can not answer that question. Please ask a more specific question or consult with a pediatric professional for personalized advice."
        )
        return
    # generate response using LLM or template
    if use_template:
        response = generate_response_with_template(query, results, document_mapping)
    else:
        # build prompt for LLM
        response = generate_response_with_llm(
            query, results, document_mapping, use_thinking=use_thinking
        )
    # post validation (how similar the response is to the query)
    embedding_response = get_embeddings([response])
    response_similarity = np.dot(query_embedding, embedding_response.T)[0][0]
    logger.info("Query-Response similarity score: %.4f", response_similarity)
    if response_similarity < ANSWER_SIMILARITY_THRESHOLD:
        logger.warning(
            "Generated response has low similarity to the query (%.4f). Consider improving the prompt or checking the retrieved documents.",
            response_similarity,
        )
        response += "\nIf a caregiver has concerns, they should speak with a pediatric professional for personalized guidance."
    print(response)


if __name__ == "__main__":
    main()
