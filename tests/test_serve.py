"""serve=: the plot in a browser, for scripts run from a terminal."""

import threading
import time
import urllib.request

import pytest

from liveplot import LivePlot
from liveplot import _serve


@pytest.fixture
def fresh_server():
    """Each test gets its own server (the real one lives for the whole process) and frame store."""
    _serve._frames = _serve._Frames()
    yield
    if _serve._server is not None:
        _serve._server.shutdown()
        _serve._server.server_close()
    _serve._server = _serve._url = None


def _get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, dict(r.headers), r.read()


def test_parse(monkeypatch):
    assert _serve.parse(False) is None
    assert _serve.parse(None) is None  # no LIVEPLOT_SERVE
    assert _serve.parse(True) == ("127.0.0.1", None)
    assert _serve.parse(9000) == ("127.0.0.1", 9000)
    assert _serve.parse("9000") == ("127.0.0.1", 9000)
    assert _serve.parse("0.0.0.0:9000") == ("0.0.0.0", 9000)
    for off in ("", "0", "false", "off"):
        monkeypatch.setenv("LIVEPLOT_SERVE", off)
        assert _serve.parse(None) is None
    monkeypatch.setenv("LIVEPLOT_SERVE", "1")
    assert _serve.parse(None) == ("127.0.0.1", None)
    monkeypatch.setenv("LIVEPLOT_SERVE", "9001")
    assert _serve.parse(None) == ("127.0.0.1", 9001)
    assert _serve.parse(False) is None, "an explicit serve=False beats the environment"


def test_page_and_long_poll(fresh_server, capsys):
    url = _serve.start("127.0.0.1", 0)
    assert "live plot at" in capsys.readouterr().err
    assert _serve.start("127.0.0.1", 0) == url, "one server per process"
    status, headers, body = _get(url)
    assert status == 200 and b"<title>liveplot</title>" in body

    got = {}
    t = threading.Thread(target=lambda: got.update(r=_get(url + "frame?after=-1")))
    t.start()
    time.sleep(0.2)
    assert t.is_alive(), "the poll waits for a frame"
    _serve.publish(b"png-0")
    t.join(5)
    status, headers, body = got["r"]
    assert (status, body, headers["X-Frame"], headers["X-Done"]) == (200, b"png-0", "0", "0")

    _serve.publish(b"png-1")
    _serve.publish(b"png-2")
    status, headers, body = _get(url + "frame?after=0")
    assert body == b"png-2", "a slow page skips straight to the newest frame"
    _serve.mark_done()
    status, headers, body = _get(url + "frame?after=" + headers["X-Frame"])
    assert body == b"png-2" and headers["X-Done"] == "1"


def test_a_script_outside_a_notebook_serves_its_frames(fresh_server, monkeypatch):
    monkeypatch.setenv("LIVEPLOT_SERVE", "127.0.0.1:0")
    plot = LivePlot("loss", total=20, refresh_seconds=0, progress=False)
    assert plot.mode == "process" and plot.serve_url
    for step in range(20):
        plot.log(step, loss=1 / (step + 1))
    plot.finish()
    status, headers, body = _get(plot.serve_url + "frame?after=-1")
    assert body == plot.last_png and body.startswith(b"\x89PNG") and headers["X-Done"] == "1"


def test_show_false_also_turns_serving_off(fresh_server):
    plot = LivePlot("loss", serve=True, show=False, progress=False)
    assert plot.mode == "off" and plot.serve_url is None and _serve._server is None


def test_a_port_in_use_warns_and_the_run_goes_on(fresh_server):
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        with pytest.warns(UserWarning, match="could not serve"):
            plot = LivePlot("loss", serve=port, progress=False)
    assert plot.serve_url is None and plot.mode == "off"
    plot.log(0, loss=1.0)
