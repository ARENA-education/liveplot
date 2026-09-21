# probe results: Google Chrome 153.0.8010.52, JupyterLab (headless), 2026-09-21 21:53:43, throttled to 1.0 Mbit/s, 40 ms (DevTools emulation)

- **FAIL** `8_backlog`: {"A": {"arrived": 50, "sent": 50, "delay_growth_ms": 16342, "max_delay_ms": 16342}, "B": {"arrived": 25, "sent": 25, "delay_growth_ms": 1042, "max_delay_ms": 1042}}

(test 8 then sent 25 kB updates at 5 Hz with a 0.8 Mbit/s budget; since rescaled to PROBE_LINK_MBPS)

# probe results: Google Chrome 153.0.8010.52, JupyterLab (headless), 2026-09-21 21:58:56, throttled to 10.0 Mbit/s, 40 ms (DevTools emulation)

- **PASS** `8_backlog`: {"A": {"arrived": 50, "sent": 50, "delay_growth_ms": 9663, "max_delay_ms": 9663}, "B": {"arrived": 15, "sent": 15, "delay_growth_ms": 367, "max_delay_ms": 403}}
