import json

from maya_csm.tags import DEFAULT_TAG_MAP, Segment, TagSpec, load_tag_map, parse_tags


def test_no_tags_single_segment_default_params():
    assert parse_tags("Hello there.") == [Segment(text="Hello there.", gen_params={})]


def test_single_leading_tag_prepends_cue_and_params():
    segs = parse_tags("[laughing] That is hilarious.")
    assert len(segs) == 1
    assert segs[0].text.startswith(DEFAULT_TAG_MAP["laughing"].cue)
    assert "That is hilarious." in segs[0].text
    assert segs[0].gen_params == DEFAULT_TAG_MAP["laughing"].gen_params
    assert segs[0].gen_params  # laughing carries a sampling override


def test_mid_text_tag_splits_into_scoped_segments():
    segs = parse_tags("I missed you. [whispering] Come closer.")
    assert len(segs) == 2
    assert segs[0] == Segment(text="I missed you.", gen_params={})
    assert segs[1].text.startswith(DEFAULT_TAG_MAP["whispering"].cue)
    assert "Come closer." in segs[1].text
    assert segs[1].gen_params == DEFAULT_TAG_MAP["whispering"].gen_params


def test_tag_scopes_until_next_tag():
    segs = parse_tags("[giggling] Stop it. [sighing] Okay, fine.")
    assert len(segs) == 2
    assert "Stop it." in segs[0].text
    assert "Okay, fine." in segs[1].text
    assert segs[0].gen_params == DEFAULT_TAG_MAP["giggling"].gen_params


def test_unknown_tag_stripped_text_kept():
    segs = parse_tags("[backflipping] Watch this.")
    assert segs == [Segment(text="Watch this.", gen_params={})]


def test_replace_position_emits_cue_inline():
    spec = DEFAULT_TAG_MAP["pause"]
    assert spec.position == "replace"
    segs = parse_tags("Wait. [pause] Never mind.")
    assert segs[1].text == f"{spec.cue} Never mind."


def test_append_position_puts_cue_after_text():
    mapping = {"sad": TagSpec(cue="Hhh...", position="append", gen_params={})}
    segs = parse_tags("[sad] I lost it.", mapping)
    assert segs[0].text == "I lost it. Hhh..."


def test_tag_only_input_yields_cue_segment():
    segs = parse_tags("[laughing]")
    assert len(segs) == 1
    assert segs[0].text == DEFAULT_TAG_MAP["laughing"].cue


def test_empty_and_whitespace_input_yield_no_segments():
    assert parse_tags("") == []
    assert parse_tags("   ") == []


def test_load_tag_map_merges_json_override(tmp_path):
    override = tmp_path / "tags.json"
    override.write_text(json.dumps({
        "laughing": {"cue": "Ha ha ha!", "position": "prepend", "gen_params": {"temperature": 1.0}},
        "purring": {"cue": "Mmm...", "position": "prepend", "gen_params": {}},
    }))
    mapping = load_tag_map(override)
    assert mapping["laughing"].cue == "Ha ha ha!"
    assert mapping["laughing"].gen_params == {"temperature": 1.0}
    assert mapping["purring"].cue == "Mmm..."
    assert "whispering" in mapping  # defaults preserved


def test_default_map_covers_goal_tags():
    for tag in ("laughing", "giggling", "whispering"):
        assert tag in DEFAULT_TAG_MAP
