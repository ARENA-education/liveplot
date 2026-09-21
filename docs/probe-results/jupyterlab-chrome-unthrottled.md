# probe results: Google Chrome 153.0.8010.52, JupyterLab (headless), 2026-09-21 20:06:28

- **PASS** `1_codecs`: {"h264": {"canPlayType": "probably", "playing": true, "frames": 109, "stutters": 0, "dropped": 71}, "vp8": {"canPlayType": "probably", "playing": true, "frames": 112, "stutters": 0, "dropped": 71}, "vp9": {"canPlayType": "probably", "playing": true, "frames": 115, "stutters": 0, "dropped": 70}}
- **PASS** `2_patch_html`: {"updates": "19/19", "videos_ms": [5, 7], "frames": 314, "stutters": 0, "max_frame_gap_ms": 50, "restarts": 0, "dropped": 0}
- **PASS** `3_patch_js`: {"updates": "19/19", "videos_ms": [3, 3], "frames": 250, "stutters": 1, "max_frame_gap_ms": 83, "restarts": 0, "dropped": 5}
- **PASS** `4_stacked`: {"frames": 272, "stutters": 0, "max_frame_gap_ms": 67, "restarts": 0, "dropped": 1}
- **PASS** `5_stress_A`: {"curve_updates": "100/100", "longest_curve_gap_ms": 312, "videos_ms": {"light_11kB": 4, "medium_166kB": 30, "heavy_2578kB": 217}, "frames": 614, "stutters": 0, "max_frame_gap_ms": 67, "restarts": 0, "dropped": 193}
- **PASS** `5_stress_B`: {"curve_updates": "100/100", "longest_curve_gap_ms": 302, "videos_ms": {}, "frames": 0, "stutters": 0, "max_frame_gap_ms": 0, "restarts": 0, "dropped": 0}
- **PASS** `6_throughput`: {"points_MB_ms": [[0.1, 18], [0.5, 72], [1, 127], [2, 200], [4, 295], [0.1, 13], [0.5, 71], [1, 129], [2, 199], [4, 282]], "fixed_cost_ms": 38, "throughput_MB_per_s": 14.9, "throughput_Mbit_per_s": 119, "r2": 0.946}

## stderr
```
[libvpx @ 0x2d617980] Bitrate not specified for constrained quality mode, using default of 256kbit/sec

```
