from maya_csm.sanitize import sanitize


def test_strips_parentheses():
    assert sanitize("hello (aside) world") == "hello aside world"


def test_strips_double_quotes_straight_and_curly():
    assert sanitize('she said "hi" and “hello”') == "she said hi and hello"


def test_semicolon_becomes_comma():
    assert sanitize("first; second") == "first, second"


def test_exclamation_removed():
    assert sanitize("wow! amazing") == "wow amazing"


def test_strips_square_brackets():
    assert sanitize("a [note] b") == "a note b"


def test_slash_becomes_space():
    assert sanitize("either/or") == "either or"


def test_collapses_whitespace():
    assert sanitize("too   many    spaces") == "too many spaces"


def test_strips_leading_and_trailing_whitespace():
    assert sanitize("  padded  ") == "padded"


def test_clean_text_passes_through():
    assert sanitize("Hey there, I am Maya.") == "Hey there, I am Maya."
