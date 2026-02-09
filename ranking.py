import logging
from typing import Dict, List, Tuple

NOISE_DEBOOST_FACTOR = 0.8


def filter_results(
    results: List[Tuple[int, float]], threshold: float
) -> List[Tuple[int, float]]:
    """
    Pre-validate the similar documents by checking similarity thresholds.
    Returns the filtered/validated list
    """
    logger = logging.getLogger("ranking.filter_results")
    valid_results: List[Tuple[int, float]] = []
    for idx, score in results:
        if score < threshold:
            logger.info(
                "Document ID %s has similarity score %.4f below threshold %.4f",
                idx,
                score,
                threshold,
            )
            continue
        valid_results.append((idx, score))
    return valid_results


def rescore_results(
    results: List[Tuple[int, float]], document_mapping: Dict[int, str]
) -> List[Tuple[int, float]]:
    """
    Re-score the results based on some heuristic or additional logic.
    This is a placeholder for demonstration purposes and can be expanded with more complex logic as needed.
    """
    logger = logging.getLogger("ranking.rescore_results")
    rescored_results: List[Tuple[int, float]] = []
    for idx, score in results:
        sentence = document_mapping.get(idx, "<missing>")
        # Example heuristic: boost score if sentence starts with "Between" or "From"
        new_score = score
        if not sentence.lower().startswith(("between", "from")):
            new_score *= NOISE_DEBOOST_FACTOR
            logger.debug(
                "Rescored Document ID %s: original score %.4f, new score %.4f",
                idx,
                score,
                new_score,
            )
        rescored_results.append((idx, new_score))
    # Sort results by new score in descending order
    rescored_results.sort(key=lambda x: x[1], reverse=True)
    return rescored_results
