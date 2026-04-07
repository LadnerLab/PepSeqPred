import importlib
import importlib.metadata as metadata

import pytest

import pepseqpred

pytestmark = pytest.mark.unit


def test_package_init_sets_unknown_version_on_missing_distribution(monkeypatch):
    original_version = metadata.version

    def _raise_not_found(_name: str) -> str:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(metadata, "version", _raise_not_found)
    reloaded = importlib.reload(pepseqpred)
    assert reloaded.__version__ == "0+unknown"

    monkeypatch.setattr(metadata, "version", original_version)
    importlib.reload(pepseqpred)
