import pytest

import liveplot.liveplot as lp


@pytest.fixture(autouse=True)
def _no_notebook_notice(monkeypatch):
    """Every test runs outside a notebook; the once-per-process notice about that is tested on its own."""
    monkeypatch.setattr(lp, "_said_nothing_is_drawn", True)
