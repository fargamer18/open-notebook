import os

import pytest

from litenotebook.config import Settings, load_settings

# Tests must never reach the network for model downloads.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture
def settings(tmp_path) -> Settings:  # type: ignore[no-untyped-def]
    """Settings rooted in a throwaway data directory."""
    return load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})
