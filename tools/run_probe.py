"""
Run a probe notebook in a real JupyterLab + Chrome, headless, and log what it measured.

    python tools/run_probe.py examples/video_probe.ipynb [--out probe-results] [--all]

Starts a JupyterLab server on a free port, opens the notebook in Chrome (Playwright's `chrome` channel,
falling back to its bundled Chromium, which can't play H.264), runs all cells while keeping the running
cell on screen (Chrome pauses muted videos scrolled out of view, which would read as stutters), then
collects every `RESULT {...}` line the notebook's outputs wrote. Writes `<out>/results.json`, a
`<out>/results.md` summary and one screenshot per cell output; exits 1 if any result is not PASS.
`--all` sets LIVEPLOT_PROBE_ALL=1 so the notebook runs even the tests its RUN table switches off.

Needs: jupyterlab, playwright (`playwright install chrome`), and the notebook's own dependencies.
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--out", default="probe-results")
    ap.add_argument("--all", action="store_true", help="run every test, ignoring the notebook's RUN table")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--throttle-mbps", type=float, default=None,
                    help="emulate a slow connection with Chrome DevTools' network throttling (download = upload = this)")
    ap.add_argument("--latency-ms", type=float, default=40, help="round-trip latency added when throttling")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="probe-"))
    (root / "nb").mkdir()
    nb = root / "nb" / Path(args.notebook).name
    shutil.copy(args.notebook, nb)
    port = free_port()
    # Render every cell (no virtual scrolling), so off-screen outputs keep their elements and scripts.
    settings = root / "settings" / "@jupyterlab" / "notebook-extension"
    settings.mkdir(parents=True)
    (settings / "tracker.jupyterlab-settings").write_text(json.dumps({"windowingMode": "none"}))
    env = {**os.environ, "JUPYTERLAB_SETTINGS_DIR": str(root / "settings"), **({"LIVEPLOT_PROBE_ALL": "1"} if args.all else {})}
    server = subprocess.Popen(
        [sys.executable, "-m", "jupyterlab", "--no-browser", f"--port={port}", "--IdentityProvider.token=",
         "--ServerApp.password=", "--allow-root", f"--ServerApp.root_dir={root / 'nb'}"],
        env=env, stdout=open(out / "server.log", "w"), stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api", timeout=1)
                break
            except OSError:
                time.sleep(1)
        with sync_playwright() as p:
            launch = dict(args=["--autoplay-policy=no-user-gesture-required", "--disable-background-timer-throttling",
                                "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows"])
            try:
                browser = p.chromium.launch(channel="chrome", **launch)
                browser_name = "Google Chrome " + browser.version
            except Exception:  # noqa: BLE001
                browser = p.chromium.launch(**launch)
                browser_name = "Chromium " + browser.version + " (no H.264)"
            page = browser.new_page(viewport={"width": 1400, "height": 1100})
            if args.throttle_mbps:
                cdp = page.context.new_cdp_session(page)
                cdp.send("Network.enable")
                rate = args.throttle_mbps * 1e6 / 8  # bytes per second
                cdp.send("Network.emulateNetworkConditions", {"offline": False, "latency": args.latency_ms,
                                                              "downloadThroughput": rate, "uploadThroughput": rate})
            page.goto(f"http://127.0.0.1:{port}/lab/tree/{nb.name}")
            page.wait_for_selector(".jp-Notebook", timeout=60000)
            time.sleep(5)
            page.click(".jp-Notebook")
            page.keyboard.press("Escape")
            page.keyboard.press("Control+Shift+c")  # command palette -> Run All Cells
            time.sleep(1)
            page.keyboard.type("Run All Cells")
            time.sleep(1)
            page.keyboard.press("Enter")
            t_end = time.time() + args.timeout
            time.sleep(3)
            while time.time() < t_end:
                running = page.locator(".jp-CodeCell.jp-mod-running, .jp-CodeCell:has(.jp-InputPrompt:text('[*]'))")
                if running.count():
                    running.first.scroll_into_view_if_needed()  # keep its videos on screen and playing
                else:
                    prompts = page.evaluate("[...document.querySelectorAll('.jp-CodeCell .jp-InputPrompt')].map(e => e.innerText.trim())")
                    if prompts and all(p_ != "[ ]:" and "*" not in p_ for p_ in prompts):
                        break
                time.sleep(1)
            time.sleep(2)
            results = []
            for text in page.evaluate("[...document.querySelectorAll('.lp-result')].map(e => e.textContent)"):
                try:
                    results.append(json.loads(text.strip().removeprefix("RESULT ")))
                except ValueError:
                    results.append({"test": "?", "verdict": "UNREADABLE", "raw": text[:300]})
            errors = page.evaluate("[...document.querySelectorAll('.jp-RenderedText[data-mime-type=\"application/vnd.jupyter.stderr\"]')].map(e => e.innerText)")
            cells = page.locator(".jp-CodeCell .jp-Cell-outputWrapper")
            for i in range(cells.count()):
                cells.nth(i).scroll_into_view_if_needed()
                cells.nth(i).screenshot(path=str(out / f"cell_{i}.png"))
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
        shutil.rmtree(root, ignore_errors=True)

    record = {"browser": browser_name, "frontend": "JupyterLab (headless)",
              "throttle": f"{args.throttle_mbps} Mbit/s, {args.latency_ms} ms (DevTools emulation)" if args.throttle_mbps else None, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
              "results": results, "stderr": errors}
    (out / "results.json").write_text(json.dumps(record, indent=2))
    lines = [f"# probe results: {browser_name}, {record['frontend']}, {record['when']}"
             + (f", throttled to {record['throttle']}" if args.throttle_mbps else ""), ""]
    lines += [f"- **{r.get('verdict')}** `{r.get('test')}`: " + json.dumps({k: v for k, v in r.items() if k not in ('test', 'verdict')})
              for r in results] or ["- no RESULT lines found (did the notebook run?)"]
    if errors:
        lines += ["", "## stderr", *[f"```\n{e}\n```" for e in errors]]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if results and all(r.get("verdict") == "PASS" for r in results) else 1)


if __name__ == "__main__":
    main()
