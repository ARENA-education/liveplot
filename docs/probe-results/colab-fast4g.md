# probe results: Colab, DevTools throttling "Fast 4G" (reported by David, 2026-09-21)

- **PASS** `5_stress_A`: {"curve_updates": "100/100", "longest_curve_gap_ms": 861, "videos_ms": {"light_10kB": 5, "medium_166kB": 28, "heavy_2578kB": 829}, "frames": 589, "stutters": 0, "max_frame_gap_ms": 51, "restarts": 0, "dropped": 0}
- **PASS** `5_stress_B`: {"curve_updates": "100/100", "longest_curve_gap_ms": 863}
- **FAIL** `6_throughput`: {"points_MB_ms": [[0.1, null], [0.5, 161], [1, 375], [2, 784], [4, 1739], [0.1, 6], [0.5, 133], [1, 339], [2, 620], [4, 959]]}

Reading: 4 MB in 1.0-1.7 s is ~25-40 Mbit/s, far above Fast 4G's ~9 Mbit/s, so DevTools throttling did not
reach Colab's output frames (cross-origin iframes, likely a separate renderer process). The first
point is null because its "start" message, sent to the same output just before the payload, never ran:
Colab can drop an update that is followed immediately by another (test 7 measures this).
