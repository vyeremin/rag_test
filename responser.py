import logging
from typing import Dict, List, Tuple


SENTENCE_CONFIDENCE_THRESHOLD = 0.85
SENTENCE_SCORE_DROP_THRESHOLD = 0.2


def generate_response_with_template(
    query: str, results: List[Tuple[int, float]], document_mapping: Dict[int, str]
) -> str:
    """
    Generate a response using a predefined template that incorporates the retrieved documents.
    This is a simplified example and can be expanded with more complex logic as needed.
    """
    logger = logging.getLogger("query.generate_response_with_template")
    response_parts = []
    top_score = 0.0
    add_concern_statement = False
    for idx, score in results:
        if top_score == 0.0:
            top_score = score
        if score < SENTENCE_CONFIDENCE_THRESHOLD:
            add_concern_statement = True
        if top_score - score > SENTENCE_SCORE_DROP_THRESHOLD:
            logger.debug(
                "Score drop detected at Document ID %s: top score %.4f, current score %.4f",
                idx,
                top_score,
                score,
            )
            break
        response_parts.append(document_mapping[idx])

    if add_concern_statement:
        response_parts.append(
            "If a caregiver has concerns, they should speak with a pediatric professional for personalized guidance."
        )
    response = f"That what I've found for you: " + " ".join(response_parts)
    logger.debug("Generated response with template:\n%s", response)
    return response
