# probe results: Google Chrome 153.0.8010.52, JupyterLab (headless), 2026-09-21 20:08:33, throttled to 10.0 Mbit/s, 40 ms (DevTools emulation)

- **PASS** `1_codecs`: {"h264": {"canPlayType": "probably", "playing": true, "frames": 115, "stutters": 0, "dropped": 71}, "vp8": {"canPlayType": "probably", "playing": true, "frames": 108, "stutters": 0, "dropped": 70}, "vp9": {"canPlayType": "probably", "playing": true, "frames": 114, "stutters": 0, "dropped": 71}}
- **PASS** `2_patch_html`: {"updates": "19/19", "videos_ms": [22, 23], "frames": 314, "stutters": 1, "max_frame_gap_ms": 100, "restarts": 0, "dropped": 0}
- **PASS** `3_patch_js`: {"updates": "19/19", "videos_ms": [23, 23], "frames": 240, "stutters": 1, "max_frame_gap_ms": 100, "restarts": 0, "dropped": 4}
- **PASS** `4_stacked`: {"frames": 279, "stutters": 0, "max_frame_gap_ms": 83, "restarts": 0, "dropped": 6}
- **PASS** `5_stress_A`: {"curve_updates": "100/100", "longest_curve_gap_ms": 4588, "videos_ms": {"light_11kB": 21, "medium_166kB": 287, "heavy_2578kB": 4477}, "frames": 612, "stutters": 0, "max_frame_gap_ms": 50, "restarts": 0, "dropped": 69}
- **PASS** `5_stress_B`: {"curve_updates": "100/100", "longest_curve_gap_ms": 4577, "videos_ms": {}, "frames": 0, "stutters": 0, "max_frame_gap_ms": 0, "restarts": 0, "dropped": 0}
- **PASS** `6_throughput`: {"points_MB_ms": [[0.1, 173], [0.5, 918], [1, 1784], [2, 3340], [4, 6706], [0.1, 11], [0.5, 875], [1, 1709], [2, 3360], [4, 6703]], "fixed_cost_ms": 8, "throughput_MB_per_s": 0.6, "throughput_Mbit_per_s": 5, "r2": 0.999}

## stderr
```
[libvpx @ 0x3ffc0980] Bitrate not specified for constrained quality mode, using default of 256kbit/sec

```
