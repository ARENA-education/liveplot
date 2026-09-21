"""
The notebook display for a LivePlot: one output holding the whole plot, laid out as matplotlib laid it
out, with each panel a separate element; and one small hidden output per panel (a "mailbox") whose
updates carry a tiny script that replaces just that panel. So a curve redraw sends only the curves, an
image only when imshow is called, a video only when a new one is logged, and a video playing in one
panel never restarts because another panel changed. No widgets, and nothing but <script> in outputs,
which JupyterLab, Notebook, Colab and VS Code all run.

What the design rests on (measured with examples/video_probe.py, results in docs/probe-results):
- Colab keeps only the newest version of an output when several updates arrive together, so every
  message is a panel's whole current state, never an event that has to arrive: skipping one is harmless.
- Outputs arrive in order and the kernel never hears when, so on a slow link a plot that sends
  everything falls further and further behind. A send budget (bytes over the last 2 s) skips curve
  redraws instead: the next one carries the newest state anyway.
- Each message carries the kernel's send time; the browser compares every arrival with the quickest one
  it has seen, which cancels the two machines' clock offset, and says so when the plot is running behind.

`finish()` writes the final state into the plot's own output as plain HTML, so a saved notebook shows
it without running anything, and empties the mailboxes.
"""

from __future__ import annotations

import base64
import collections
import threading
import time
import uuid
import zlib

_WINDOW = 2.0  # seconds of history the send budget looks at
_BEHIND_MS = 2000  # show "N s behind" once the plot runs this far behind real time
_STYLE = "position:absolute;margin:0;padding:0;border:0;max-width:none"


def _pct(v, total):
    return f"{100 * v / total:.4f}%"


def _box_style(rect, size):
    x, y, w, h = rect
    width, height = size
    return f"left:{_pct(x, width)};top:{_pct(y, height)};width:{_pct(w, width)};height:{_pct(h, height)}"


class HtmlDisplay:
    def __init__(self, max_mbps: float | None = 4.0):
        from IPython.display import HTML, display

        self.uid = uuid.uuid4().hex[:10]
        self.host = display(HTML("<i>live plot: waiting for the first frame…</i>"), display_id=True)
        self.boxes = {}  # ("tile" | "video", panel) -> display handle of that panel's mailbox
        self.budget = max_mbps * 1e6 / 8 * _WINDOW if max_mbps else None  # bytes per window
        self.window = collections.deque()  # (time, bytes) of recent sends
        self.lock = threading.RLock()
        self.frame = None  # the newest frame the renderer produced
        self.layout = None  # (layout_id, size) the host output was built for
        self.sent = {}  # panel -> checksum of the tile its mailbox last carried
        self.pending = set()  # panels whose newest tile hasn't gone out (over budget)
        self.videos = {}  # panel -> (data, mime) of its newest video
        self.video_sent = {}  # panel -> id of the video its mailbox last carried
        self.bytes_sent = collections.Counter()  # "host" / ("tile", panel) / ("video", panel) -> bytes, for anyone curious

    # -- called on the notebook's main thread ---------------------------------------------------------

    def ensure_boxes(self, n_panels: int, video_panels=()):
        """Create the mailboxes the panels need. New outputs must be made on the main thread, while the
        cell runs (from a background thread they could land in whichever cell runs next)."""
        from IPython.display import HTML, display

        keys = [("tile", i) for i in range(n_panels)] + [("video", i) for i in video_panels]
        for key in keys:
            if key not in self.boxes:
                self.boxes[key] = display(HTML(""), display_id=True)

    # -- called from the frame collector / encoder threads ---------------------------------------------

    def show(self, frame: dict):
        """A new frame from the renderer: rebuild the plot's output if the layout changed, else queue the
        tiles that changed and send what the budget allows."""
        with self.lock:
            self.frame = frame
            layout = (frame["layout_id"], frame["size"], tuple(sorted(frame["tiles"])))
            if layout != self.layout:
                self._send_host(frame)
                self.layout = layout
                return
            for i, (rect, png) in frame["tiles"].items():
                if self.sent.get(i) != (rect, zlib.crc32(png)):
                    self.pending.add(i)
            self.flush()

    def flush(self):
        """Send the queued tiles the budget allows (called on every frame and every half second)."""
        with self.lock:
            if self.frame is None or self.layout is None:
                return
            for i in sorted(self.pending):
                if ("tile", i) not in self.boxes or i not in self.frame["tiles"]:
                    continue
                rect, png = self.frame["tiles"][i]
                js = self._tile_js(i, rect, png)
                if i in self.sent and not self._fits(len(js)):
                    continue  # over budget: the next frame's tile replaces this one anyway
                self._post(("tile", i), js)
                self.sent[i] = (rect, zlib.crc32(png))
                self.pending.discard(i)

    def video(self, panel: int, data: bytes, mime: str):
        """A new video for a panel: always sent (it counts against the budget, so curves ease off after)."""
        with self.lock:
            self.videos[panel] = (data, mime)
            if ("video", panel) not in self.boxes or self.layout is None:
                return  # the next host rebuild carries it
            self._post(("video", panel), self._status_js(panel, f"new video ({len(data) / 1e3:.0f} kB): downloading…"))
            self._post(("video", panel), self._video_js(panel, data, mime))
            self.video_sent[panel] = id(data)

    def video_status(self, panel: int, text: str):
        """A line of text over a video panel (e.g. why there is no video), or "" to clear it."""
        with self.lock:
            if ("video", panel) in self.boxes and self.layout is not None:
                self._post(("video", panel), self._status_js(panel, text))

    def finish(self):
        """The final state, written into the plot's own output as plain HTML; the mailboxes emptied."""
        from IPython.display import HTML

        with self.lock:
            if self.frame is not None:
                self._send_host(self.frame, static=True)
            for handle in self.boxes.values():
                handle.update(HTML(""))

    # -- plumbing ---------------------------------------------------------------------------------------

    def _fits(self, nbytes: int) -> bool:
        if self.budget is None:
            return True
        now = time.monotonic()
        while self.window and now - self.window[0][0] > _WINDOW:
            self.window.popleft()
        # nothing sent for a whole window: send, however big (else a tile bigger than the budget would never go)
        return not self.window or sum(b for _, b in self.window) + nbytes <= self.budget

    def _post(self, key, js: str):
        from IPython.display import HTML

        self.window.append((time.monotonic(), len(js)))
        self.bytes_sent[key] += len(js)
        self.boxes[key].update(HTML(f"<script>{js}</script>"))

    def _preamble(self) -> str:
        """Find the plot's output and note how late this message is (for the "running behind" line)."""
        return (
            f'var root = document.getElementById("lp-{self.uid}"); if (!root) return;'
            f'var S = window["lp_{self.uid}"] || (window["lp_{self.uid}"] = {{quickest: Infinity}});'
            f"var late = Date.now() - {time.time() * 1000:.0f}; S.quickest = Math.min(S.quickest, late);"
            f'var behind = late - S.quickest, note = root.querySelector("[data-lp=behind]");'
            f'if (note) note.textContent = behind > {_BEHIND_MS} ? "plot " + Math.round(behind / 1000) + " s behind: slow connection" : "";'
        )

    def _tile_js(self, i, rect, png) -> str:
        size = self.frame["size"]
        video = ""
        if i in self.frame["videos"]:  # keep the player on the picture if the panel moved
            video = f'var v = root.querySelector("[data-lp=video-{i}]"); if (v) v.style.cssText = "{_STYLE};{_box_style(self.frame["videos"][i], size)}";'
        return (
            f"(function() {{ {self._preamble()}"
            f'var el = root.querySelector("[data-lp=tile-{i}]");'
            f'if (!el) {{ el = document.createElement("img"); el.setAttribute("data-lp", "tile-{i}"); root.insertBefore(el, root.firstChild); }}'
            f'el.style.cssText = "{_STYLE};{_box_style(rect, size)}";'
            f'el.src = "data:image/png;base64,{base64.b64encode(png).decode()}"; {video} }})();'
        )

    def _status_js(self, panel, text) -> str:
        text = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'(function() {{ {self._preamble()} var s = root.querySelector("[data-lp=status-{panel}]"); if (s) s.textContent = "{text}"; }})();'

    def _video_js(self, panel, data, mime) -> str:
        return (
            f"(function() {{ {self._preamble()}"
            f'var v = root.querySelector("[data-lp=video-{panel}]"), s = root.querySelector("[data-lp=status-{panel}]");'
            f'if (!v) return; v.src = "data:{mime};base64,{base64.b64encode(data).decode()}";'
            f'if (s) s.textContent = ""; }})();'
        )

    def _send_host(self, frame, static=False):
        """The whole plot as one output: every tile, and a player on each video panel's picture."""
        from IPython.display import HTML

        width, height = frame["size"]
        parts = [
            f'<div id="lp-{self.uid}" style="position:relative;width:{width}px;max-width:100%;'
            f'aspect-ratio:{width}/{height};background:white;overflow:hidden">'
        ]
        for i, (rect, png) in sorted(frame["tiles"].items()):
            parts.append(f'<img data-lp="tile-{i}" style="{_STYLE};{_box_style(rect, frame["size"])}" '
                         f'src="data:image/png;base64,{base64.b64encode(png).decode()}">')
            self.sent[i] = (rect, zlib.crc32(png))
        for i, rect in sorted(frame["videos"].items()):
            src = ""
            if i in self.videos:
                data, mime = self.videos[i]
                src = f' src="data:{mime};base64,{base64.b64encode(data).decode()}"'
                self.video_sent[i] = id(data)
            parts.append(f'<video data-lp="video-{i}" autoplay loop muted playsinline style="{_STYLE};'
                         f'{_box_style(rect, frame["size"])};object-fit:contain"{src}></video>')
            x, y, _, _ = rect
            if not static:
                parts.append(f'<div data-lp="status-{i}" style="position:absolute;left:{_pct(x, width)};top:{_pct(y, height)};'
                             f'font:11px monospace;color:#333;background:rgba(255,255,255,0.8);padding:0 3px"></div>')
        if not static:
            parts.append('<div data-lp="behind" style="position:absolute;right:4px;top:2px;font:11px monospace;'
                         'color:#b00;background:rgba(255,255,255,0.85);padding:0 3px"></div>')
        parts.append("</div>")
        html = "".join(parts)
        self.pending.clear()
        self.bytes_sent["host"] += len(html)
        self.window.append((time.monotonic(), len(html)))
        self.host.update(HTML(html))
