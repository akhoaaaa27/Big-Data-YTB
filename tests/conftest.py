import pytest

from ytb import config


@pytest.fixture(autouse=True)
def isolate_data(tmp_path, monkeypatch):
    """Không ghi vào dữ liệu thật hoặc dùng lại dữ liệu của test trước."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "STAGING_DIR", tmp_path / "staging")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
