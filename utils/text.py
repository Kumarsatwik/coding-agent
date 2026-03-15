"""
Token counting and estimation utilities.

Used for API cost tracking, context window management, and rate limiting.
"""
import tiktoken


def get_tokenizer(model: str):
    """Get tokenizer for model, fallback to cl100k_base."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return encoding.encode
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
        return encoding.encode


def count_tokens(text: str, model: str) -> int:
    """Count tokens using model tokenizer, fallback to estimation."""
    tokenizer = get_tokenizer(model)

    try:
        return len(tokenizer(text))
    except Exception:
        return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    """Estimate tokens from characters (1 token ≈ 4 chars)."""
    return max(1, len(text) // 4)
