"""
Serve the live plot to a browser, for scripts run from a terminal.

One small HTTP server per Python process, on a daemon thread, shared by every LivePlot made with
`serve=`: the page always shows the newest frame of the most recent plot, so a script that trains
several runs one after another keeps one URL. Standard library only.

    GET /                 the page
    GET /frame?after=N    the newest frame once its number is greater than N (long poll, ~25 s cap);
                          headers X-Frame (its number) and X-Done (1 once the plot finished)
"""

from __future__ import annotations

import http.server
import os
import socket
import sys
import threading

DEFAULT_PORT = 8765
_PORTS_TO_TRY = 20  # 8765..8784, then whatever the OS gives us

_PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>liveplot</title>
<style>
  :root { color-scheme: light dark; --bg: #fff; --fg: #555; }
  @media (prefers-color-scheme: dark) { :root { --bg: #1e1e1e; --fg: #aaa; } }
  body { margin: 0; padding: 16px; background: var(--bg); color: var(--fg);
         font: 13px system-ui, sans-serif; }
  img { display: block; max-width: 100%; height: auto; background: #fff; }
  #status { margin-bottom: 8px; }
</style></head>
<body><div id="status">waiting for the first frame&hellip;</div><img id="plot" alt="">
<script>
const img = document.getElementById("plot"), status = document.getElementById("status");
let after = -1, url = null;
async function poll() {
  for (;;) {
    try {
      const r = await fetch("frame?after=" + after, {cache: "no-store"});
      if (r.status === 200) {
        after = parseInt(r.headers.get("X-Frame"));
        const next = URL.createObjectURL(await r.blob());
        img.src = next;
        if (url) URL.revokeObjectURL(url);
        url = next;
        const done = r.headers.get("X-Done") === "1";
        status.textContent = (done ? "finished" : "live") + " \\u00b7 frame " + after +
                             " \\u00b7 " + new Date().toLocaleTimeString();
      }
    } catch (e) {
      status.textContent = "disconnected: the script has ended or the server is unreachable" +
                           (after >= 0 ? " (last frame kept)" : "");
      await new Promise(res => setTimeout(res, 2000));
    }
  }
}
poll();
</script></body></html>
"""


class _Frames:
    """The newest frame, its number, and a condition to wait on for the next one."""

    def __init__(self):
        self.cond = threading.Condition()
        self.png: bytes | None = None
        self.n = -1
        self.done = False

    def publish(self, png: bytes, done: bool = False):
        with self.cond:
            self.png, self.done = png, done
            self.n += 1
            self.cond.notify_all()

    def mark_done(self):
        with self.cond:
            if self.png is not None and not self.done:
                self.done = True
                self.n += 1  # waiting pages get the same image again, now labelled finished
                self.cond.notify_all()

    def wait_newer(self, after: int, timeout: float):
        with self.cond:
            self.cond.wait_for(lambda: self.n > after, timeout=timeout)
            return self.n, self.png, self.done


_frames = _Frames()
_server = None
_url = None
_lock = threading.Lock()


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server's name
        path, _, query = self.path.partition("?")
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", _PAGE)
        elif path == "/frame":
            try:
                after = int(dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("after", -1))
            except ValueError:
                after = -1
            n, png, done = _frames.wait_newer(after, timeout=25.0)
            if png is None or n <= after:
                self._send(204, "text/plain", b"")  # nothing new yet: the page simply asks again
            else:
                self._send(200, "image/png", png, {"X-Frame": str(n), "X-Done": "1" if done else "0"})
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code, ctype, body, headers=None):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the tab was closed mid-poll

    def log_message(self, *args):  # don't write a line per request into the training output
        pass


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True  # a long poll in flight must not keep the script alive at exit
    allow_reuse_address = True


def parse(serve) -> tuple[str, int | None] | None:
    """`serve` (or $LIVEPLOT_SERVE when it is None) -> (host, port or None for the default), or None for off."""
    if serve is None:
        serve = os.environ.get("LIVEPLOT_SERVE", "")
        if serve.strip().lower() in ("", "0", "false", "no", "off"):
            return None
        if serve.strip().lower() in ("1", "true", "yes", "on"):
            serve = True
    if serve is False:
        return None
    if serve is True:
        return "127.0.0.1", None
    if isinstance(serve, int):
        return "127.0.0.1", serve
    host, sep, port = str(serve).strip().rpartition(":")
    if not sep:  # "8000"
        return "127.0.0.1", int(port)
    return (host or "127.0.0.1"), int(port)


def start(host: str, port: int | None) -> str:
    """Start the process's server if it isn't running (later calls reuse it) and return its URL."""
    global _server, _url
    with _lock:
        if _server is not None:
            return _url
        ports = [port] if port is not None else [*range(DEFAULT_PORT, DEFAULT_PORT + _PORTS_TO_TRY), 0]
        err = None
        for p in ports:
            try:
                _server = _Server((host, p), _Handler)
                break
            except OSError as e:
                err = e
        else:
            raise err
        threading.Thread(target=_server.serve_forever, name="liveplot-serve", daemon=True).start()
        bound = _server.server_address[1]
        shown = "localhost" if host in ("127.0.0.1", "localhost") else (socket.gethostname() if host in ("0.0.0.0", "") else host)
        _url = f"http://{shown}:{bound}/"
        _announce(_url, host, bound)
        return _url


def _announce(url, host, port):
    msg = f"liveplot: live plot at {url}"
    if host in ("127.0.0.1", "localhost"):
        msg += f"   (on a remote machine: ssh -L {port}:localhost:{port} <host>; VS Code forwards it for you)"
    try:
        from tqdm import tqdm

        tqdm.write(msg, file=sys.stderr)  # lands above any bar that is already running
    except ImportError:
        print(msg, file=sys.stderr, flush=True)


def publish(png: bytes):
    _frames.publish(png)


def mark_done():
    _frames.mark_done()
