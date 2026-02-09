import logging
from typing import Dict, List, Tuple


def build_prompt(
    query: str, results: List[Tuple[int, float]], document_mapping: Dict[int, str]
) -> Dict[str, str]:
    """
    Build a prompt for the LLM based on the query and retrieved context.
    """
    logger = logging.getLogger("query.build_prompt")
    system_prompt = f"""You are a helpful assistant for answering questions about early childhood development based on provided information. Carefully read the provided information and answer the QUESTION at the end. Follow these guidelines when formulating your answer:
1. Use provided information to answer the question.
2. If the question is too vague, inappropriate or offensive - refuse answering the question OR provide safe uncertain statement OR a generic guidance.
3. If the question is about some uncommon, unusual or abnormal situation - strongly suggest to speak with a pediatric professional.
4. If can not figure out answer based on provided information - response politely that you can not answer the question.
5. Start directly with the answer - DO NOT ADD **The answer** or similar phrases at the beginning of the answer.
6. Answer should be in plain text without markdown or any formatting (including numbering).
7. Prioritize SAFETY. It is okay to not answer the question if answer cannot be confirmed with confidence.

Here is the information you have retrieved:

"""
    user_prompt = "\n".join(
        # f"Document {idx}.\n{document_mapping.get(idx, '<missing>')}"
        f"{document_mapping.get(idx, '<missing>')}"
        for idx, score in results
    )
    user_prompt = system_prompt + user_prompt + f"\n\nQUESTION:\n{query}"

    messages = [{"role": "user", "content": user_prompt}]
    logger.debug("Constructed user prompt for LLM:\n%s", user_prompt)
    return messages
