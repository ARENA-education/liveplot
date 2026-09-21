# probe results: Colab, DevTools throttling "Fast 4G" (reported by David, 2026-09-21)

- **PASS** `5_stress_A`: {"curve_updates": "100/100", "longest_curve_gap_ms": 861, "videos_ms": {"light_10kB": 5, "medium_166kB": 28, "heavy_2578kB": 829}, "frames": 589, "stutters": 0, "max_frame_gap_ms": 51, "restarts": 0, "dropped": 0}
- **PASS** `5_stress_B`: {"curve_updates": "100/100", "longest_curve_gap_ms": 863}
- **FAIL** `6_throughput`: {"points_MB_ms": [[0.1, null], [0.5, 161], [1, 375], [2, 784], [4, 1739], [0.1, 6], [0.5, 133], [1, 339], [2, 620], [4, 959]]}

Reading: 4 MB in 1.0-1.7 s is ~25-40 Mbit/s, far above Fast 4G's ~9 Mbit/s, so DevTools throttling did not
reach Colab's output frames (cross-origin iframes, likely a separate renderer process). The first
point is null because its "start" message, sent to the same output just before the payload, never ran:
Colab can drop an update that is followed immediately by another (test 7 measures this).

# Colab, unthrottled, tests 6 and 7 (reported by David, 2026-09-21)

- **PASS** `6_throughput`: {"points_MB_ms": [[0.1, "lost"], [0.5, 205], [1, 340], [2, 591], [4, 1058], [0.1, 2], [0.5, 139], [1, 180], [2, 662], [4, 1013]], "fixed_cost_ms": 30, "throughput_MB_per_s": 3.9, "throughput_Mbit_per_s": 31, "r2": 0.969}
- **PASS** `7_bursts`: {"gap_0ms": "1/20", "gap_10ms": "20/20", "gap_50ms": "20/20", "gap_200ms": "20/20", "big_update_survived_a_follower": true, "lost_at_0ms": [0..18]}

Reading: Colab renders only the newest version of an output when several arrive together (19 of 20
back-to-back updates never ran; at 10 ms spacing all did). JupyterLab never drops any. So every
liveplot message must be a panel's whole current state, one hidden output per panel, never events
that must all arrive. Colab's pipeline carried ~31 Mbit/s of payload on a 400 Mbit/s line, 30 ms fixed.
