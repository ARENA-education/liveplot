# %% [markdown]
# # liveplot video probe
#
# Can this frontend (Colab, VS Code, JupyterLab, ...) do what a live plot with a video panel needs?
# Run all cells. Each test writes a `RESULT {...}` line into its own output, with a verdict (`PASS` /
# `FAIL`) and the measurements behind it; a screenshot of those lines is all that's needed.
#
# | test | what it checks |
# |---|---|
# | 1 | which video codecs this frontend can play |
# | 2 | a `<script>` in one output can update another output of the same cell, beside a playing video |
# | 3 | the same, sent as `IPython.display.Javascript` |
# | 4 | the no-script fallback: two stacked outputs updated independently |
# | 5 | big video deliveries don't stall the curves (two plots at once) |
# | 6 | throughput calibration: how long payloads of known size take to arrive |
# | 7 | whether back-to-back updates of one output get lost |
# | 8 | whether a slow link makes the plot fall further behind, and whether a send budget stops it |
#
# Tests that have already passed on the frontends we care about are switched off in `RUN` below, so
# a re-run only does the new ones. Set `LIVEPLOT_PROBE_ALL=1` in the kernel's environment to run all.
# Video playback is measured by the browser itself: every frame it presents
# (`requestVideoFrameCallback`), frames it dropped (`getVideoPlaybackQuality`), and jumps back to the
# start that aren't the clip looping or a new video (restarts).
#
# Nothing here imports liveplot: it tests only the notebook frontend.

# %%
import os
import subprocess
import sys

RUN = {
    "1_codecs": False,  # passed: JupyterLab + Chrome, Colab
    "2_patch_html": False,  # passed: JupyterLab + Chrome, Colab
    "3_patch_js": False,  # passed: JupyterLab + Chrome, Colab
    "4_stacked": False,  # passed: JupyterLab + Chrome, Colab
    "5_stress": False,  # passed: JupyterLab + Chrome (also throttled to 10 Mbit/s), Colab. Turn on for a throttled run
    "6_throughput": False,  # passed: JupyterLab + Chrome, Colab (~31 Mbit/s; DevTools throttling doesn't reach Colab)
    "7_bursts": False,  # JupyterLab: never loses one. Colab: keeps only the newest of back-to-back updates
    "8_backlog": True,
}
if os.environ.get("LIVEPLOT_PROBE_ALL"):
    RUN = dict.fromkeys(RUN, True)
print("running:", [name for name, on in RUN.items() if on])

try:
    import imageio_ffmpeg  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imageio-ffmpeg"], check=True)

import base64
import io
import tempfile
import time
import uuid

import imageio_ffmpeg
import numpy as np
from IPython.display import HTML, Javascript, display
from PIL import Image, ImageDraw


def ball_frames(n=90, w=320, h=240, colour=(40, 90, 200), flash_every=45):
    """A ball bouncing across a white frame, with a pink flash every `flash_every` frames (an RL reset)."""
    frames = np.full((n, h, w, 3), 255, np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    for t in range(n):
        if t % flash_every < 5:
            frames[t] = (255, 200, 200)
        x = 20 + (w - 40) * abs((t / (n / 2)) % 2 - 1)
        y = h / 2 + (h / 3) * np.sin(2 * np.pi * t / n * 2)
        frames[t][(xx - x) ** 2 + (yy - y) ** 2 < 15**2] = colour
    return frames


def noisy_frames(n, amount, w=320, h=240, seed=0):
    """The ball clip with `amount` of per-pixel noise mixed in: 0 = light, 1 = incompressible static."""
    rng = np.random.default_rng(seed)
    base = ball_frames(n=n, w=w, h=h, colour=(40, 160, 60)).astype(np.float32)
    noise = rng.integers(0, 256, size=base.shape).astype(np.float32)
    return ((1 - amount) * base + amount * noise).astype(np.uint8)


def encode(frames, codec="h264", fps=30):
    """(T, H, W, 3) uint8 -> encoded bytes, with the ffmpeg that imageio-ffmpeg bundles."""
    suffix, lib, params = {
        "h264": (".mp4", "libx264", ["-movflags", "+faststart"]),
        "vp8": (".webm", "libvpx", ["-b:v", "0", "-crf", "30"]),
        "vp9": (".webm", "libvpx-vp9", ["-b:v", "0", "-crf", "40", "-deadline", "realtime", "-cpu-used", "8"]),
    }[codec]
    path = tempfile.mktemp(suffix=suffix)
    h, w = frames.shape[1:3]
    writer = imageio_ffmpeg.write_frames(path, (w, h), fps=fps, codec=lib, output_params=params, pix_fmt_out="yuv420p")
    writer.send(None)
    for f in frames:
        writer.send(np.ascontiguousarray(f))
    writer.close()
    data = open(path, "rb").read()
    os.remove(path)
    return data


def data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def counter_png(i, w=320, h=240):
    """A stand-in for liveplot's curves panel: a PNG that visibly changes on every update."""
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.line([(x, h / 2 - 60 * np.sin(x / 25 + i / 2)) for x in range(0, w, 4)], fill=(31, 119, 180), width=3)
    d.text((10, 10), f"curves update {i}", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# Browser-side helpers, pasted into each output's <script>: Colab runs every output in its own sandboxed
# frame, so nothing can be shared through `window` across outputs of different cells.
WATCH_JS = """
function lpWatch(v, st) {
  // Playback health of <video> v, kept in st: frames presented, stutters (a gap over 2.5 frame
  // intervals), the longest gap, restarts (jumping back that isn't the clip looping or a new video),
  // and the browser's own dropped-frame count. Set st.swapAt = performance.now() when changing src on
  // purpose: anything within half a second of that is the swap, not a stutter or a restart.
  st.frames = 0; st.stutters = 0; st.max_frame_gap_ms = 0; st.restarts = 0; st.dropped = 0; st.swapAt = 0;
  function swapping() { return performance.now() - st.swapAt < 500; }
  var lastNow = null, lastT = 0, armed = false, expected = 1000 / 30;
  function onFrame(now) {
    armed = false;
    if (lastNow !== null && !swapping()) {
      var g = now - lastNow;
      st.max_frame_gap_ms = Math.max(st.max_frame_gap_ms, Math.round(g));
      if (g > 2.5 * expected) st.stutters += 1;
    }
    lastNow = now; st.frames += 1; arm();
  }
  function arm() { if (!armed && v.requestVideoFrameCallback) { armed = true; v.requestVideoFrameCallback(onFrame); } }
  v.addEventListener("playing", function() { lastNow = null; arm(); });
  arm();
  setInterval(function() {
    var looped = v.duration && lastT > v.duration - 0.3;
    if (v.currentTime + 0.05 < lastT && !looped && !swapping()) st.restarts += 1;
    lastT = v.currentTime;
    var q = v.getVideoPlaybackQuality ? v.getVideoPlaybackQuality() : null;
    if (q) st.dropped = q.droppedVideoFrames;
  }, 100);
}
function lpResult(el, test, verdict, data) {
  el.innerText = "RESULT " + JSON.stringify(Object.assign({test: test, verdict: verdict}, data));
  el.style.color = verdict === "PASS" ? "green" : (verdict === "FAIL" ? "red" : "gray");
}
"""


def result_line(uid, test):
    return f'<pre class="lp-result" id="res-{uid}" style="font-size:11px;white-space:pre-wrap">RESULT {{"test": "{test}", "verdict": "FAIL", "why": "the browser never ran this output\'s script"}}</pre>'


# %% [markdown]
# ## Test 1: which codecs play here
# Three copies of the same 3-second clip; each should show a moving ball.

# %%
def test_codecs():
    uid = uuid.uuid4().hex[:8]
    frames = ball_frames()
    videos = {codec: encode(frames, codec) for codec in ("h264", "vp8", "vp9")}
    mime = {"h264": "video/mp4", "vp8": "video/webm", "vp9": "video/webm"}
    types = {"h264": 'video/mp4; codecs="avc1.42E01E"', "vp8": 'video/webm; codecs="vp8"', "vp9": 'video/webm; codecs="vp9"'}
    cells = "".join(
        f'<div style="text-align:center"><b>{c}</b> ({len(d) / 1e3:.0f} kB)<br>'
        f'<video id="v-{uid}-{c}" autoplay loop muted playsinline width="200" src="{data_uri(d, mime[c])}"></video></div>'
        for c, d in videos.items()
    )
    display(HTML(f"""<div style="display:flex;gap:16px;font-family:monospace">{cells}</div>{result_line(uid, "1_codecs")}
<script>{WATCH_JS}
(function() {{
  var codecs = {list(types)}, types = {types}, st = {{}};
  codecs.forEach(function(c) {{ st[c] = {{}}; lpWatch(document.getElementById("v-{uid}-" + c), st[c]); }});
  setTimeout(function() {{
    var out = {{}}, ok = true;
    codecs.forEach(function(c) {{
      var v = document.getElementById("v-{uid}-" + c);
      out[c] = {{canPlayType: v.canPlayType(types[c]) || "no", playing: v.currentTime > 0, frames: st[c].frames, stutters: st[c].stutters, dropped: st[c].dropped}};
    }});
    ok = out.h264.playing;
    lpResult(document.getElementById("res-{uid}"), "1_codecs", ok ? "PASS" : "FAIL", out);
  }}, 4000);
}})();
</script>"""))
    time.sleep(5)  # keep the cell busy while the browser measures


if RUN["1_codecs"]:
    test_codecs()

# %% [markdown]
# ## Tests 2 and 3: one output patches another, beside a playing video
# The first output is the "plot": a curves image and a video side by side. The second output is a
# "mailbox", replaced every half second with a tiny script that swaps only the curves image. Twice, it
# announces a new video and then delivers it. PASS means every update arrived, the video never
# restarted except when a new one came, and playback didn't stutter.

# %%
def test_patching(via):
    test = "2_patch_html" if via == "html" else "3_patch_js"
    uid = uuid.uuid4().hex[:8]
    display(HTML(f"""
<div style="display:flex;gap:16px;align-items:flex-start;font-family:monospace;font-size:12px">
  <div><div>curves</div><img id="img-{uid}" src="{data_uri(counter_png(0), 'image/png')}" width="320"></div>
  <div><div id="vstat-{uid}">video: none yet</div>
       <video id="vid-{uid}" autoplay loop muted playsinline width="320" style="background:#eee;height:240px"></video></div>
</div>{result_line(uid, test)}
<script>{WATCH_JS}
(function() {{
  window["lp_{uid}"] = {{updates: 0, videos: [], st: {{}}}};
  lpWatch(document.getElementById("vid-{uid}"), window["lp_{uid}"].st);
  var timer = setInterval(function() {{
    // measured only while the test runs: the kernel sets s.final at the end, and the verdict freezes
    var s = window["lp_{uid}"], st = s.st;
    var ok = s.updates >= 19 && s.videos.length === 2 && st.restarts === 0 && st.stutters <= 2;
    lpResult(document.getElementById("res-{uid}"), "{test}", s.final ? (ok ? "PASS" : "FAIL") : "running",
      {{updates: s.updates + "/19", videos_ms: s.videos, frames: st.frames, stutters: st.stutters, max_frame_gap_ms: st.max_frame_gap_ms, restarts: st.restarts, dropped: st.dropped}});
    if (s.final) clearInterval(timer);
  }}, 250);
}})();
</script>"""))
    mailbox = display(HTML("<i style='font-size:10px'>mailbox</i>"), display_id=True)

    def send(js):
        js = f'(function() {{ var s = window["lp_{uid}"]; if (!s) return; {js} }})();'
        mailbox.update(HTML(f"<i style='font-size:10px'>mailbox</i><script>{js}</script>") if via == "html" else Javascript(js))

    for i in range(1, 20):
        send(f'document.getElementById("img-{uid}").src = "{data_uri(counter_png(i), "image/png")}"; s.updates += 1;')
        if i in (4, 12):
            data = encode(ball_frames(colour=(40, 90, 200) if i == 4 else (200, 40, 40)))
            send(f's.t0 = performance.now(); document.getElementById("vstat-{uid}").innerText = "new video ({len(data) / 1e3:.0f} kB), downloading...";')
            send(f"""var dt = Math.round(performance.now() - s.t0); s.videos.push(dt); s.st.swapAt = performance.now();
                     document.getElementById("vid-{uid}").src = "{data_uri(data, 'video/mp4')}";
                     document.getElementById("vstat-{uid}").innerText = "video " + s.videos.length + " ({len(data) / 1e3:.0f} kB) arrived in " + dt + " ms";""")
        time.sleep(0.5)
    time.sleep(2)  # let the last video play a little before the verdict freezes
    send("s.final = true;")
    time.sleep(0.5)


if RUN["2_patch_html"]:
    test_patching("html")

# %%
if RUN["3_patch_js"]:
    test_patching("javascript")

# %% [markdown]
# ## Test 4: no scripts at all (the fallback)
# Two stacked outputs: a "curves" image replaced every half second, and a video below it, set once. The
# measuring script below the video only observes; the mechanism itself uses no JavaScript.

# %%
def test_stacked():
    uid = uuid.uuid4().hex[:8]
    curves = display(HTML(f'<img src="{data_uri(counter_png(0), "image/png")}" width="320">'), display_id=True)
    display(HTML(f"""<video id="vid-{uid}" autoplay loop muted playsinline width="320" src="{data_uri(encode(ball_frames()), "video/mp4")}"></video>
{result_line(uid, "4_stacked")}
<script>{WATCH_JS}
(function() {{
  var st = {{}}, t0 = performance.now();
  lpWatch(document.getElementById("vid-{uid}"), st);
  var timer = setInterval(function() {{
    var done = performance.now() - t0 > 9500, ok = st.frames > 150 && st.restarts === 0 && st.stutters <= 2;
    lpResult(document.getElementById("res-{uid}"), "4_stacked", done ? (ok ? "PASS" : "FAIL") : "running",
      {{frames: st.frames, stutters: st.stutters, max_frame_gap_ms: st.max_frame_gap_ms, restarts: st.restarts, dropped: st.dropped}});
    if (done) clearInterval(timer);
  }}, 250);
}})();
</script>"""))
    for i in range(1, 20):
        curves.update(HTML(f'<img src="{data_uri(counter_png(i), "image/png")}" width="320">'))
        time.sleep(0.5)
    time.sleep(1)


if RUN["4_stacked"]:
    test_stacked()

# %% [markdown]
# ## Test 5: do video deliveries stall the curves?
# Two plots in one cell, each with curves updating 5 times a second for 20 seconds. Plot A receives
# light, medium and noise-heavy (~2.6 MB, deliberately abusive) videos; plot B none. PASS means both
# got all 100 curve updates and A's video never stuttered or restarted except at a new video.

# %%
def test_stress():
    videos = {name: encode(noisy_frames(90, amount)) for name, amount in (("light", 0.0), ("medium", 0.15), ("heavy", 1.0))}
    print({name: f"{len(v) / 1e3:.0f} kB" for name, v in videos.items()})
    boxes = {}
    for label in ("A", "B"):
        uid = uuid.uuid4().hex[:8]
        display(HTML(f"""<div style="font-family:monospace;font-size:12px"><b>plot {label}</b>
<div style="display:flex;gap:16px;align-items:flex-start">
  <img id="img-{uid}" src="{data_uri(counter_png(0), 'image/png')}" width="240">
  <div><div id="vstat-{uid}">video: none</div>
       <video id="vid-{uid}" autoplay loop muted playsinline width="240" style="background:#eee;height:180px"></video></div>
</div>{result_line(uid, "5_stress_" + label)}</div>
<script>{WATCH_JS}
(function() {{  // its own scope: two plots on one page must not share variables
  window["lp_{uid}"] = {{arrivals: [], videos: {{}}, st: {{}}}};
  lpWatch(document.getElementById("vid-{uid}"), window["lp_{uid}"].st);
  var timer = setInterval(function() {{
    var s = window["lp_{uid}"], a = s.arrivals, gap = 0, st = s.st;
    for (var i = 1; i < a.length; i++) gap = Math.max(gap, a[i] - a[i - 1]);
    var ok = a.length >= 100 && st.restarts === 0 && st.stutters <= 3;
    if (s.final) clearInterval(timer);
    lpResult(document.getElementById("res-{uid}"), "5_stress_{label}", s.final ? (ok ? "PASS" : "FAIL") : "running",
      {{curve_updates: a.length + "/100", longest_curve_gap_ms: Math.round(gap), videos_ms: s.videos, frames: st.frames, stutters: st.stutters, max_frame_gap_ms: st.max_frame_gap_ms, restarts: st.restarts, dropped: st.dropped}});
  }}, 250);
}})();
</script>"""))
        boxes[label] = (uid, display(HTML("<i style='font-size:10px'>mailbox</i>"), display_id=True))

    def post(label, js):
        uid, box = boxes[label]
        box.update(HTML(f'<i style="font-size:10px">mailbox</i><script>(function() {{ var s = window["lp_{uid}"]; if (!s) return; {js} }})();</script>'))

    ua = boxes["A"][0]
    sent = []
    for i in range(1, 101):
        for label, (uid, _) in boxes.items():
            post(label, f'document.getElementById("img-{uid}").src = "{data_uri(counter_png(i), "image/png")}"; s.arrivals.push(performance.now());')
        if i in (20, 50, 80):
            name = {20: "light", 50: "medium", 80: "heavy"}[i]
            data = videos[name]
            t0 = time.time()
            post("A", f's.t0 = performance.now(); document.getElementById("vstat-{ua}").innerText = "{name} video ({len(data) / 1e3:.0f} kB): downloading...";')
            post("A", f"""var dt = Math.round(performance.now() - s.t0); s.videos["{name}_{len(data) // 1000}kB"] = dt; s.st.swapAt = performance.now();
                          document.getElementById("vid-{ua}").src = "{data_uri(data, 'video/mp4')}";
                          document.getElementById("vstat-{ua}").innerText = "{name} video ({len(data) / 1e3:.0f} kB): arrived in " + dt + " ms";""")
            sent.append(f"{name} ({len(data) / 1e3:.0f} kB): {(time.time() - t0) * 1000:.0f} ms")
        time.sleep(0.2)
    time.sleep(2)
    for label in boxes:
        post(label, "s.final = true;")
    time.sleep(0.5)
    print("kernel side, handing each video to the network:", "; ".join(sent))


if RUN["5_stress"]:
    test_stress()

# %% [markdown]
# ## Test 6: throughput calibration
# Payloads of random (incompressible) bytes, 0.1 to 4 MB, each sent twice, timed from a tiny "announce"
# message to the payload's script running. A straight line through (size, time) separates the fixed cost
# per message (latency plus the browser parsing and running it: the intercept) from the transfer itself
# (the slope, i.e. the real throughput to this browser). PASS means the line fits (R^2 > 0.9), which
# means the "arrived in ... ms" numbers elsewhere are measuring transfer and not an artefact.

# %%
def test_throughput():
    uid = uuid.uuid4().hex[:8]
    display(HTML(f"""<div id="tp-{uid}" style="font-family:monospace;font-size:12px">throughput: waiting</div>{result_line(uid, "6_throughput")}
<script>{WATCH_JS}
(function() {{
  window["lp_{uid}"] = {{points: []}};
  setInterval(function() {{
    var all = window["lp_{uid}"].points, p = all.filter(function(q) {{ return isFinite(q[1]); }}), n = p.length;
    if (n < 2) return;
    var mx = 0, my = 0; p.forEach(function(q) {{ mx += q[0] / n; my += q[1] / n; }});
    var sxy = 0, sxx = 0, syy = 0; p.forEach(function(q) {{ sxy += (q[0] - mx) * (q[1] - my); sxx += (q[0] - mx) ** 2; syy += (q[1] - my) ** 2; }});
    var slope = sxy / sxx, icpt = my - slope * mx, r2 = sxy * sxy / (sxx * syy);
    var done = all.length >= 10, ok = done && r2 > 0.9 && slope > 0;
    lpResult(document.getElementById("res-{uid}"), "6_throughput", ok ? "PASS" : (done ? "FAIL" : "running"),
      {{points_MB_ms: all.map(function(q) {{ return [q[0], isFinite(q[1]) ? Math.round(q[1]) : "lost"]; }}), fixed_cost_ms: Math.round(icpt),
        throughput_MB_per_s: +(1000 / slope).toFixed(1), throughput_Mbit_per_s: +(8000 / slope).toFixed(0), r2: +r2.toFixed(3)}});
  }}, 250);
}})();
</script>"""))
    box = display(HTML("<i style='font-size:10px'>mailbox</i>"), display_id=True)

    def post(js):
        box.update(HTML(f'<i style="font-size:10px">mailbox</i><script>(function() {{ var s = window["lp_{uid}"]; if (!s) return; {js} }})();</script>'))

    for mb in (0.1, 0.5, 1, 2, 4) * 2:
        payload = base64.b64encode(os.urandom(int(mb * 1e6))).decode()
        post(f"s.t0 = performance.now();")
        post(f'var x = "{payload}"; s.points.push([{mb}, performance.now() - s.t0 - (x.length ? 0 : 1)]);')
        time.sleep(1.0)
    time.sleep(1)


if RUN["6_throughput"]:
    test_throughput()

# %% [markdown]
# ## Test 7: are back-to-back updates of one output ever lost?
# liveplot sends each change as a new version of one small output. If the frontend only renders the
# newest version when several arrive together, the ones in between never run. That's harmless for curves
# (the next update replaces them) but not for a video. Bursts of 20 updates are sent with 0, 10, 50 and
# 200 ms between them, and the page counts how many ran. Then a 1 MB update is followed at once by a tiny
# one: did the big one run? PASS means nothing was lost at 50 ms spacing and the big update survived;
# losses at 0-10 ms are reported, not failed.

# %%
def test_bursts():
    uid = uuid.uuid4().hex[:8]
    gaps = (0, 10, 50, 200)
    display(HTML(f"""<div id="b-{uid}" style="font-family:monospace;font-size:12px">bursts: waiting</div>{result_line(uid, "7_bursts")}
<script>{WATCH_JS}
(function() {{
  window["lp_{uid}"] = {{got: {{}}, big: false, final: false}};
  var timer = setInterval(function() {{
    var s = window["lp_{uid}"], counts = {{}};
    {list(gaps)}.forEach(function(g) {{ counts["gap_" + g + "ms"] = (s.got[g] || []).length + "/20"; }});
    var ok = (s.got[50] || []).length === 20 && (s.got[200] || []).length === 20 && s.big;
    lpResult(document.getElementById("res-{uid}"), "7_bursts", s.final ? (ok ? "PASS" : "FAIL") : "running",
      Object.assign(counts, {{big_update_survived_a_follower: s.big, lost_at_0ms: (function() {{
        var got = s.got[0] || [], lost = []; for (var i = 0; i < 20; i++) if (got.indexOf(i) < 0) lost.push(i); return lost; }})()}}));
    if (s.final) clearInterval(timer);
  }}, 250);
}})();
</script>"""))
    box = display(HTML("<i style='font-size:10px'>mailbox</i>"), display_id=True)

    def post(js):
        box.update(HTML(f'<i style="font-size:10px">mailbox</i><script>(function() {{ var s = window["lp_{uid}"]; if (!s) return; {js} }})();</script>'))

    for gap in gaps:
        for i in range(20):
            post(f"(s.got[{gap}] = s.got[{gap}] || []).push({i});")
            if gap:
                time.sleep(gap / 1000)
        time.sleep(1)
    post(f'var x = "{base64.b64encode(os.urandom(1_000_000)).decode()}"; s.big = x.length > 0;')
    post("s.after_big = true;")
    time.sleep(3)
    post("s.final = true;")
    time.sleep(1)


if RUN["7_bursts"]:
    test_bursts()

# %% [markdown]
# ## Test 8: does a slow link make the plot fall further and further behind?
# The kernel sends updates 5 times a second for 10 seconds, sized so they total 1.3x the link speed
# assumed in PROBE_LINK_MBPS (incompressible data, a stand-in for curve images). Each carries the kernel's send time; the browser measures each
# arrival's delay *relative to the first one* (so the two machines' clocks needn't agree). Phase A sends
# everything; phase B skips an update whenever the last 2 seconds' bytes would exceed a budget, so the
# newest state goes out and stale ones never do. On a link slower than the send rate, A's delay should
# grow steadily and B's should stay flat. Run it on a throttled connection to see the difference;
# unthrottled, both stay flat.

# %%
PROBE_LINK_MBPS = 10  # the connection we're designing for; throttle to this to see phase A fall behind
PROBE_BUDGET_MBPS = 0.4 * PROBE_LINK_MBPS  # phase B's send budget, with room left for videos and everything else


def test_backlog():
    uid = uuid.uuid4().hex[:8]
    display(HTML(f"""<div style="font-family:monospace;font-size:12px">
  <div id="lag-{uid}">delay: waiting</div></div>{result_line(uid, "8_backlog")}
<script>{WATCH_JS}
(function() {{
  window["lp_{uid}"] = {{A: [], B: [], final: false}};
  var timer = setInterval(function() {{
    var s = window["lp_{uid}"], out = {{}};
    ["A", "B"].forEach(function(ph) {{
      var d = s[ph];
      if (!d.length) return;
      var rel = d.map(function(q) {{ return q[1] - d[0][1]; }});
      out[ph] = {{arrived: d.length, sent: s["sent_" + ph], delay_growth_ms: Math.round(rel[rel.length - 1]),
                  max_delay_ms: Math.round(Math.max.apply(null, rel))}};
    }});
    document.getElementById("lag-{uid}").innerText = "delay relative to the first update: " + JSON.stringify(out);
    var ok = out.B && out.B.delay_growth_ms < 1000;
    lpResult(document.getElementById("res-{uid}"), "8_backlog", s.final ? (ok ? "PASS" : "FAIL") : "running", out);
    if (s.final) clearInterval(timer);
  }}, 250);
}})();
</script>"""))
    box = display(HTML("<i style='font-size:10px'>mailbox</i>"), display_id=True)

    def post(js):
        box.update(HTML(f'<i style="font-size:10px">mailbox</i><script>(function() {{ var s = window["lp_{uid}"]; if (!s) return; {js} }})();</script>'))

    for phase, budget in (("A", None), ("B", PROBE_BUDGET_MBPS)):
        sent, window_ = 0, []  # window_: (time, bytes) of recent sends, for the budget
        raw = int(1.3 * PROBE_LINK_MBPS * 1e6 / 8 * 0.2 * 3 / 4)  # bytes per update, before base64
        for i in range(50):
            payload = base64.b64encode(os.urandom(raw)).decode()
            now = time.time()
            window_ = [(t, b) for t, b in window_ if now - t < 2.0]
            if budget is not None and sum(b for _, b in window_) + len(payload) > budget * 1e6 / 8 * 2.0:
                time.sleep(0.2)
                continue  # skip: the next update carries the newest state anyway
            post(f'var x = "{payload}"; s.{phase}.push([{i}, Date.now() - {now * 1000:.0f}]);')
            window_.append((now, len(payload)))
            sent += 1
            time.sleep(0.2)
        post(f"s.sent_{phase} = {sent};")
        time.sleep(20)  # let whatever is still in the pipe arrive before the next phase
    post("s.final = true;")
    time.sleep(1)


if RUN["8_backlog"]:
    test_backlog()
