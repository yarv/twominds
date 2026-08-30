"""Embedding-backend input bounds (no API calls)."""

from twominds.embed import OpenAIEmbedder


def test_truncate_bounds_long_inputs_only():
    short = "a short response"
    # ~50k tokens of ordinary prose — far past the 8192-token API cap
    long = "the quick brown fox jumps over the lazy dog " * 5000
    out_short, out_long = OpenAIEmbedder._truncate([short, long])
    assert out_short == short  # short inputs pass through untouched
    assert len(out_long) < len(long)
    # 8000 tokens is at most ~8000 * 6 chars for prose (tiktoken path) and
    # 8000 bytes on the offline fallback — either way far below the original
    assert len(out_long) <= OpenAIEmbedder._MAX_INPUT_TOKENS * 6
    assert long.startswith(out_long[:100])  # a prefix, not garbage


def test_truncate_treats_special_token_text_as_text():
    # A model may emit the literal text of a tokenizer special token; it is
    # ordinary data here and must neither raise nor be dropped.
    weird = "The answer is no. <|fim_suffix|> <|endoftext|> trailing words"
    (out,) = OpenAIEmbedder._truncate([weird])
    assert out == weird


def test_truncate_bounds_token_dense_text():
    # emoji are token-dense (multiple tokens per char): a char-count heuristic
    # would miss this; the token/byte bound must still catch it
    dense = "🐍🦊🐺" * 20000
    (out,) = OpenAIEmbedder._truncate([dense])
    assert len(out) < len(dense)
