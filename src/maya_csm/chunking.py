"""Sentence-aware text splitting.

csm-maya-exp2 degrades on long inputs (the author's demo caps at 200 chars);
each chunk becomes one generate() call and the audio is joined afterwards.
"""

import re

_SENTENCE_RE = re.compile(r"(?<=[.?…])\s+")


def _split_words(sentence: str, max_chars: int) -> list[str]:
    chunks, current = [], ""
    for word in sentence.split():
        if current and len(current) + 1 + len(word) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        chunks.append(current)
    return chunks


def split_text(text: str, max_chars: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks, current = [], ""
    for sentence in _SENTENCE_RE.split(text):
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_words(sentence, max_chars))
        elif current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks
