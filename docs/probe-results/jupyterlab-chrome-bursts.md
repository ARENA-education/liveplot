# probe results: Google Chrome 153.0.8010.52, JupyterLab (headless), 2026-09-21 20:28:21

- **PASS** `5_stress_A`: {"curve_updates": "100/100", "longest_curve_gap_ms": 322, "videos_ms": {"light_11kB": 4, "medium_166kB": 28, "heavy_2578kB": 227}, "frames": 617, "stutters": 1, "max_frame_gap_ms": 83, "restarts": 0, "dropped": 1}
- **PASS** `5_stress_B`: {"curve_updates": "100/100", "longest_curve_gap_ms": 318, "videos_ms": {}, "frames": 0, "stutters": 0, "max_frame_gap_ms": 0, "restarts": 0, "dropped": 0}
- **PASS** `6_throughput`: {"points_MB_ms": [[0.1, 18], [0.5, 74], [1, 127], [2, 186], [4, 304], [0.1, 19], [0.5, 68], [1, 123], [2, 184], [4, 285]], "fixed_cost_ms": 36, "throughput_MB_per_s": 14.8, "throughput_Mbit_per_s": 118, "r2": 0.968}
- **PASS** `7_bursts`: {"gap_0ms": "20/20", "gap_10ms": "20/20", "gap_50ms": "20/20", "gap_200ms": "20/20", "big_update_survived_a_follower": true, "lost_at_0ms": []}
