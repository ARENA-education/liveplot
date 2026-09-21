# %% [markdown]
# # liveplot display check
#
# Runs the synthetic RL demo (curves beside a looping rollout video, `examples/rl_video_synthetic.py`)
# and watches liveplot's output from the outside, in the browser: how many curve updates arrived, how
# many videos were swapped in, whether the video ever restarted or stuttered in between, whether the plot
# fell behind, and whether the final state still plays after `finish()`. It writes one `RESULT {...}`
# line with a verdict; run all cells and read (or screenshot) that line.

# %%
import os
import subprocess
import sys
import urllib.request

BRANCH = "video-dev"
try:
    import google.colab  # noqa: F401

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imageio-ffmpeg",
                    f"git+https://github.com/ARENA-education/liveplot.git@{BRANCH}"], check=True)
except ImportError:
    pass
try:
    import rl_video_synthetic
except ImportError:  # not run from the repo's examples/ folder: fetch the demo next to this notebook
    url = f"https://raw.githubusercontent.com/ARENA-education/liveplot/{BRANCH}/examples/rl_video_synthetic.py"
    urllib.request.urlretrieve(url, "rl_video_synthetic.py")
    sys.path.insert(0, os.getcwd())
    import rl_video_synthetic

import time
import uuid

from IPython.display import HTML, display

import liveplot

print("liveplot from", liveplot.__file__)

# %%
WATCH_JS = """
(function() {
  // Watch the live plot of this cell from outside: tile updates, video swaps, restarts, stutters, "behind".
  var S = window.lpCheck = {tiles: 0, swaps: 0, frames: 0, stutters: 0, max_gap_ms: 0, restarts: 0,
                            behind_max_s: 0, final: false};
  var seen = new WeakSet(), lastSrc = new WeakMap(), video = null, lastNow = null, lastT = 0, swapAt = 0;
  function watchVideo(v) {
    if (v === video) return;
    video = v; lastNow = null; lastT = 0;
    function onFrame(now) {
      if (lastNow !== null && performance.now() - swapAt > 500) {
        var g = now - lastNow; S.max_gap_ms = Math.max(S.max_gap_ms, Math.round(g)); if (g > 100) S.stutters += 1;  // a visible hitch
      }
      lastNow = now; S.frames += 1; if (video === v) v.requestVideoFrameCallback(onFrame);
    }
    if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(onFrame);
  }
  setInterval(function() {
    var root = document.querySelector('[id^="lp-"]');
    if (!root) return;
    root.querySelectorAll('img[data-lp^="tile-"]').forEach(function(img) {
      if (lastSrc.get(img) !== img.src) { if (lastSrc.has(img)) S.tiles += 1; lastSrc.set(img, img.src); }
    });
    var v = root.querySelector('video[data-lp^="video-"]');
    if (v) {
      if (lastSrc.get(v) !== v.currentSrc && v.currentSrc) { if (lastSrc.has(v)) { S.swaps += 1; } swapAt = performance.now(); lastSrc.set(v, v.currentSrc); lastT = 0; }
      watchVideo(v);
      var looped = v.duration && lastT > v.duration - 0.3;
      if (v.currentTime + 0.05 < lastT && !looped && performance.now() - swapAt > 500) S.restarts += 1;
      lastT = v.currentTime;
    }
    var behind = root.querySelector('[data-lp="behind"]');
    var m = behind && behind.textContent.match(/plot (\\d+) s behind/);
    if (m) S.behind_max_s = Math.max(S.behind_max_s, +m[1]);
  }, 50);
})();
"""


def check(args, expect_tiles, expect_videos):
    uid = uuid.uuid4().hex[:8]
    display(HTML(f'<pre class="lp-result" id="res-{uid}" style="font-size:11px;white-space:pre-wrap">'
                 f'RESULT {{"test": "display", "verdict": "FAIL", "why": "the watcher never ran"}}</pre><script>{WATCH_JS}</script>'))
    report = display(HTML(""), display_id=True)
    t0 = time.time()
    plot = rl_video_synthetic.train(args)
    seconds = time.time() - t0
    time.sleep(3)  # let the final (static) state play a little
    report.update(HTML(f"""<script>(function() {{
      var S = window.lpCheck, root = document.querySelector('[id^="lp-"]'), v = root && root.querySelector("video");
      var t1 = v ? v.currentTime : -1;
      setTimeout(function() {{
        var playing = v && v.currentTime !== t1 && !!v.currentSrc, static_ = root && !root.querySelector('[data-lp="behind"]');
        var ok = S.tiles >= {expect_tiles} && S.swaps >= {expect_videos} && S.restarts === 0 && S.stutters <= 3 && playing && static_;
        var el = document.getElementById("res-{uid}");
        el.textContent = "RESULT " + JSON.stringify({{test: "display", verdict: ok ? "PASS" : "FAIL", curve_tile_updates: S.tiles,
          video_swaps: S.swaps, video_frames: S.frames, stutters: S.stutters, max_frame_gap_ms: S.max_gap_ms, restarts: S.restarts,
          behind_max_s: S.behind_max_s, final_video_plays: playing, final_state_static: static_,
          kernel_bytes_sent: {dict((str(k), v) for k, v in plot._html.bytes_sent.items()) if plot._html else {}},
          run_seconds: {seconds:.1f}}});
        el.style.color = ok ? "green" : "red";
      }}, 1500);
    }})();</script>"""))
    time.sleep(2)


# %% [markdown]
# ## The check
# 40 updates at 0.6 s (the course's pace), a new rollout video every 10 updates. Expect PASS: at least
# 20 curve updates seen, 3 video swaps (4 videos), no restarts, playback without stutters, and a final
# state that keeps playing with nothing left running.

# %%
check(rl_video_synthetic.Args(), expect_tiles=20, expect_videos=3)
