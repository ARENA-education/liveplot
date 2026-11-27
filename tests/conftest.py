import pytest


@pytest.fixture(autouse=True)
def _no_serving_from_the_environment(monkeypatch):
    """LIVEPLOT_SERVE in the developer's shell must not turn every plot in the suite into a server."""
    monkeypatch.delenv("LIVEPLOT_SERVE", raising=False)
