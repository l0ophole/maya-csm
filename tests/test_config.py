from maya_csm.config import Settings


def test_defaults():
    s = Settings.from_env(env={})
    assert s.base_model == "sesame/csm-1b"
    assert s.speaker_id == "4"
    assert s.max_chunk_chars == 150
    assert s.gap_ms == 120
    assert s.sampling is True
    assert s.reference_clips == []
    assert "csm-maya-exp2" in s.adapter_path
    assert s.dtype == "float16"
    assert s.devices == ()
    assert s.compile is False


def test_env_overrides():
    s = Settings.from_env(env={
        "MAYA_BASE_MODEL": "unsloth/csm-1b",
        "MAYA_ADAPTER": "shb777/csm-maya-exp2",
        "MAYA_MAX_CHUNK_CHARS": "120",
        "MAYA_GAP_MS": "80",
        "MAYA_SAMPLING": "0",
        "HF_TOKEN": "hf_xxx",
    })
    assert s.base_model == "unsloth/csm-1b"
    assert s.adapter_path == "shb777/csm-maya-exp2"
    assert s.max_chunk_chars == 120
    assert s.gap_ms == 80
    assert s.sampling is False
    assert s.hf_token == "hf_xxx"


def test_perf_env_overrides():
    s = Settings.from_env(env={
        "MAYA_DTYPE": "bfloat16",
        "MAYA_DEVICES": "cuda:0,cuda:1",
        "MAYA_COMPILE": "1",
    })
    assert s.dtype == "bfloat16"
    assert s.devices == ("cuda:0", "cuda:1")
    assert s.compile is True


def test_devices_env_is_whitespace_tolerant_and_optional():
    assert Settings.from_env(env={"MAYA_DEVICES": " cuda:0 , cuda:1 "}).devices == ("cuda:0", "cuda:1")
    assert Settings.from_env(env={"MAYA_DEVICES": ""}).devices == ()
    assert Settings.from_env(env={}).devices == ()


def test_compile_env_is_falsey_by_default():
    assert Settings.from_env(env={"MAYA_COMPILE": "0"}).compile is False
    assert Settings.from_env(env={"MAYA_COMPILE": "no"}).compile is False


def test_tag_map_env_points_to_json(tmp_path):
    override = tmp_path / "tags.json"
    override.write_text('{"purring": {"cue": "Mmm..."}}')
    s = Settings.from_env(env={"MAYA_TAG_MAP": str(override)})
    assert "purring" in s.tag_map
    assert "laughing" in s.tag_map
