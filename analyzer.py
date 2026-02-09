import re


def cleanup_query(text: str) -> str:
    """
    Cleans up input text by stripping unwanted characters,
    normalizing whitespace, and removing HTML tags.
    """
    if not text:
        return ""

    # 1. Strip Unwanted Characters
    # Keeps only alphanumeric, spaces, and basic punctuation (.,!?-).
    # Removes emojis, symbols like @#$%, and special control characters.
    text = re.sub(r"[^\w\s\.\!\?\,\-]", "", text)

    # 2. Normalize Whitespace
    # Collapses multiple spaces, tabs, or newlines into a single space.
    text = re.sub(r"\s+", " ", text)

    # 3. Basic Sanitization (Remove HTML tags)
    # Strips out anything inside angle brackets < >.
    text = re.sub(r"<[^>]*>", "", text)

    # Final trim of leading/trailing whitespace
    return text.strip()
