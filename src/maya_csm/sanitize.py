"""Character sanitization for csm-maya-exp2.

The model mispronounces or garbles ( ) " " ; ! [ ] / (per the model README's
known issues); these rules mirror the author's own preprocessing in the
TinkerSpace demo. Must run AFTER tag extraction, since it strips brackets.
"""

import re

_REMOVE = "()“”\"[]"
_TRANSLATE = str.maketrans({**{c: "" for c in _REMOVE}, ";": ",", "!": " ", "/": " "})


def sanitize(text: str) -> str:
    text = text.translate(_TRANSLATE)
    return re.sub(r"\s+", " ", text).strip()
