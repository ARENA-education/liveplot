# %% [markdown]
# # liveplot video probe
#
# Can this frontend (Colab, VS Code, JupyterLab, ...) do what a live plot with a video panel needs?
# Run the cells top to bottom (Runtime -> Run all). Each test prints **PASS** / **FAIL** lines into its
# own output; a screenshot of the whole notebook afterwards is all that's needed.
#
# | test | what it checks | why liveplot cares |
# |---|---|---|
# | 1 | which video codecs this frontend can play | picks the codec liveplot encodes with |
# | 2 | a script in one output can reach another output of the same cell | lets curves and video update independently, side by side (option 2) |
# | 3 | the same, with `IPython.display.Javascript` instead of `<script>` in HTML | a second route to the same thing |
# | 4 | two stacked outputs updated independently, no scripts at all | the fallback if 2 and 3 fail (option 1) |
#
# Nothing here imports liveplot: it tests only the notebook frontend.

# %%
# Setup: imageio-ffmpeg brings its own ffmpeg (with H.264 and VP8/VP9 encoders), so nothing else is needed.
import subprocess
import sys

try:
    import imageio_ffmpeg  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imageio-ffmpeg"], check=True)

import base64
import io
import os
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


def encode(frames, codec, fps=30):
    """(T, H, W, 3) uint8 -> encoded bytes, with the ffmpeg that imageio-ffmpeg bundles."""
    suffix, params = {
        "h264": (".mp4", ["-movflags", "+faststart"]),
        "vp8": (".webm", ["-b:v", "0", "-crf", "30"]),
        "vp9": (".webm", ["-b:v", "0", "-crf", "40", "-deadline", "realtime", "-cpu-used", "8"]),
    }[codec]
    lib = {"h264": "libx264", "vp8": "libvpx", "vp9": "libvpx-vp9"}[codec]
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
    pts = [(x, h / 2 - 60 * np.sin(x / 25 + i / 2)) for x in range(0, w, 4)]
    d.line(pts, fill=(31, 119, 180), width=3)
    d.text((10, 10), f"curves update {i}", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


MIME = {"h264": "video/mp4", "vp8": "video/webm", "vp9": "video/webm"}
frames = ball_frames()
videos = {codec: encode(frames, codec) for codec in ("h264", "vp8", "vp9")}
print({codec: f"{len(v) / 1e3:.0f} kB" for codec, v in videos.items()}, "-", imageio_ffmpeg.get_ffmpeg_version())

# %% [markdown]
# ## Test 1: which codecs play here
# Three copies of the same 3-second clip. Each should show a moving ball; the line under each says
# whether the browser says it can play it and whether it actually started playing.

# %%
uid = uuid.uuid4().hex[:8]
cells = []
for codec, data in videos.items():
    mime = MIME[codec]
    cells.append(f"""
    <div style="text-align:center">
      <div><b>{codec}</b> ({len(data) / 1e3:.0f} kB)</div>
      <video id="v-{uid}-{codec}" autoplay loop muted playsinline width="240" src="{data_uri(data, mime)}"></video>
      <div id="s-{uid}-{codec}">(scripts did not run)</div>
    </div>""")
types = {"h264": 'video/mp4; codecs="avc1.42E01E"', "vp8": 'video/webm; codecs="vp8"', "vp9": 'video/webm; codecs="vp9"'}
script = "".join(
    f"""
  (function() {{
    var v = document.getElementById("v-{uid}-{codec}"), s = document.getElementById("s-{uid}-{codec}");
    var can = v.canPlayType('{types[codec]}') || "no";
    s.innerText = "canPlayType: " + can + " / waiting to play...";
    var done = false;
    v.addEventListener("playing", function() {{ if (!done) {{ done = true; s.innerText = "canPlayType: " + can + " / PASS: plays"; }} }});
    v.addEventListener("error", function() {{ s.innerText = "canPlayType: " + can + " / FAIL: " + (v.error ? "error code " + v.error.code : "error"); }});
    setTimeout(function() {{ if (!done && !(v.currentTime > 0)) s.innerText = "canPlayType: " + can + " / FAIL: never started"; else if (!done) s.innerText = "canPlayType: " + can + " / PASS: plays"; }}, 4000);
  }})();"""
    for codec in types
)
display(HTML(f'<div style="display:flex;gap:16px;font-family:monospace">{"".join(cells)}</div><script>{script}</script>'))

# %% [markdown]
# ## Test 2: one output patches another (`<script>` in HTML)
# The first output is the "plot": a curves image and a video side by side. The second output is a
# "mailbox" that gets replaced every half second with a tiny script which swaps only the curves image.
# Twice, it announces a new video, then delivers it; the status line above the video shows how long
# the delivery took. **Watch the video: it should never jump back to the start when the curves update**,
# only when a new video arrives (the second video is red instead of blue).
#
# Expect: the curves count up to 19, the report line says `PASS`, and the video status ends with
# `video 2 ... playing`.

# %%
def probe_patching(uid, via):
    """`via`: "html" sends each update as <script> inside HTML; "javascript" as IPython.display.Javascript."""
    first = data_uri(counter_png(0), "image/png")
    display(HTML(f"""
<div style="display:flex;gap:16px;align-items:flex-start;font-family:monospace;font-size:12px">
  <div><div>curves</div><img id="img-{uid}" src="{first}" width="320"></div>
  <div><div id="vstat-{uid}">video: none yet</div>
       <video id="vid-{uid}" autoplay loop muted playsinline width="320" style="background:#eee;height:240px"></video>
       <div id="vtime-{uid}">(host scripts did not run)</div></div>
</div>
<div id="report-{uid}" style="font-family:monospace">report: no mailbox message has reached this output yet</div>
<script>
  (function() {{
    var v = document.getElementById("vid-{uid}"), t = document.getElementById("vtime-{uid}"), last = 0, restarts = 0;
    window["lp_restarts_{uid}"] = 0;
    setInterval(function() {{
      // going backwards is a restart, unless it's the normal loop at the end of the clip or a new video
      var looped = v.duration && last > v.duration - 0.3;
      if (v.currentTime + 0.05 < last && !looped && !window["lp_swapping_{uid}"]) {{ restarts += 1; window["lp_restarts_{uid}"] = restarts; }}
      window["lp_swapping_{uid}"] = false;
      last = v.currentTime;
      t.innerText = "video time " + v.currentTime.toFixed(2) + " s, unexpected restarts: " + restarts;
    }}, 100);
  }})();
</script>"""))
    mailbox = display(HTML("<i>mailbox</i>"), display_id=True)

    def send(js):
        if via == "html":
            mailbox.update(HTML(f"<i style='font-size:10px'>mailbox</i><script>{js}</script>"))
        else:
            mailbox.update(Javascript(js))

    def find(name):
        return f'document.getElementById("{name}-{uid}")'

    colours = {1: (40, 90, 200), 2: (200, 40, 40)}
    for i in range(1, 20):
        send(f"""
          var img = {find('img')}, rep = {find('report')};
          if (!img) {{ console.log("liveplot probe: host not found"); }}
          else {{ img.src = "{data_uri(counter_png(i), 'image/png')}";
                  rep.innerText = "report: PASS - mailbox update " + {i} + " reached the plot's output"; }}""")
        if i in (4, 12):
            n = 1 if i == 4 else 2
            data = encode(ball_frames(colour=colours[n]), "h264")
            send(f"""
              window["lp_t0_{uid}"] = performance.now();
              var s = {find('vstat')}; if (s) s.innerText = "video {n}: new video logged ({len(data) / 1e3:.0f} kB), downloading...";""")
            send(f"""
              var v = {find('vid')}, s = {find('vstat')};
              if (v) {{
                var dt = performance.now() - (window["lp_t0_{uid}"] || performance.now());
                window["lp_swapping_{uid}"] = true;
                v.src = "{data_uri(data, 'video/mp4')}";
                s.innerText = "video {n}: received in " + dt.toFixed(0) + " ms ({len(data) / 1e3:.0f} kB), starting...";
                v.addEventListener("playing", function() {{ s.innerText = "video {n}: received in " + dt.toFixed(0) + " ms ({len(data) / 1e3:.0f} kB), playing"; }}, {{once: true}});
              }}""")
        time.sleep(0.5)
    print("done; if the report line above still says 'no mailbox message', this route FAILs in this frontend")


probe_patching(uuid.uuid4().hex[:8], via="html")

# %% [markdown]
# ## Test 3: the same with `IPython.display.Javascript`
# Same expectations as test 2.

# %%
probe_patching(uuid.uuid4().hex[:8], via="javascript")

# %% [markdown]
# ## Test 4: no scripts at all (option 1)
# Two stacked outputs: a "curves" image replaced every half second, and a video below it set once.
# The video should play on without restarting while the image above it updates. This works without any
# JavaScript, but the video sits under the plot, not beside it.

# %%
curves = display(HTML(f'<img src="{data_uri(counter_png(0), "image/png")}" width="320">'), display_id=True)
display(HTML(f'<video autoplay loop muted playsinline width="320" src="{data_uri(videos["h264"], "video/mp4")}"></video>'))
for i in range(1, 20):
    curves.update(HTML(f'<img src="{data_uri(counter_png(i), "image/png")}" width="320">'))
    time.sleep(0.5)
print("done: the curves above counted to 19 while the video kept playing")
