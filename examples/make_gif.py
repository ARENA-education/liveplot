"""Re-create docs/demo.gif by running examples/demo.py with recording on (works as a plain script)."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / "examples")]
import demo  # noqa: E402

out = root / "docs" / "demo.gif"
plot = demo.train_classifier(pace_seconds=0.01, record=str(out), refresh_seconds=0.12, cell_size=(4.2, 3.0), dpi=80, progress=False)
print(f"{len(plot.frames)} frames -> {out} ({out.stat().st_size // 1024} KiB); final: { {k: round(v, 3) for k, v in plot.latest.items()} }")
