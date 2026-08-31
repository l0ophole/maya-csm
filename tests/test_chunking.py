from maya_csm.chunking import split_text


def test_short_text_passes_through():
    assert split_text("Hello there.", max_chars=150) == ["Hello there."]


def test_splits_on_sentence_boundaries():
    text = "First sentence here. Second sentence here. Third sentence here."
    chunks = split_text(text, max_chars=45)
    assert len(chunks) > 1
    assert all(len(c) <= 45 for c in chunks)
    assert " ".join(chunks) == text


def test_packs_multiple_sentences_per_chunk_when_they_fit():
    text = "One. Two. Three."
    assert split_text(text, max_chars=150) == ["One. Two. Three."]


def test_long_sentence_falls_back_to_word_split():
    text = "word " * 60  # a single 300-char "sentence" with no punctuation
    chunks = split_text(text.strip(), max_chars=100)
    assert len(chunks) >= 3
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks) == text.strip()


def test_empty_input_returns_empty_list():
    assert split_text("", max_chars=100) == []
    assert split_text("   ", max_chars=100) == []
