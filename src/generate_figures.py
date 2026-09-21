"""Export README figures to reports/figures/ (CC6).

Four PNGs, all computed from the same functions the pipeline and app use (no
separate figure-only logic that could quietly drift from the real numbers):

1. dispute_rate_trend.png       -- raw vs. corrected monthly dispute rate
2. dispute_rate_anomaly.png     -- the rolling-baseline anomaly check that
                                    flags August
3. top_segment_drivers.png      -- top count drivers of the corrected spike
4. app_screenshot.png           -- the Streamlit app, captured headless via
                                    the system Chrome (no new browser-
                                    automation dependency)

Run with::

    python src/generate_figures.py
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PROJECT_ROOT
from driver_analysis import build_driver_report
from metric_engine import monthly_trend_table, resolve_tables
from quality_checks import (
    DEFAULT_THRESHOLDS,
    TABLE_TRANSACTIONS,
    monthly_dispute_rate_series,
    rolling_baseline,
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# A small, print-friendly palette reused across all three data charts so a
# reader learns "orange = raw / reported, blue = corrected" once (matching
# app/streamlit_app.py's COLOR_RAW/COLOR_CORRECTED convention).
COLOR_RAW = "#eb6834"
COLOR_CORRECTED = "#2f6fed"
COLOR_ANOMALY = "#c0392b"
COLOR_NEUTRAL = "#4a4a4a"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#888888",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "font.size": 10,
        "legend.frameon": False,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
    }
)


def chart_dispute_rate_trend() -> Path:
    tables = resolve_tables()
    trend = monthly_trend_table(tables[TABLE_TRANSACTIONS])

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(
        trend["month"], trend["dispute_rate_raw"] * 100, marker="o", color=COLOR_RAW, label="Raw (reported)"
    )
    ax.plot(
        trend["month"],
        trend["dispute_rate_corrected"] * 100,
        marker="o",
        color=COLOR_CORRECTED,
        label="Corrected (deduplicated)",
    )
    ax.set_title("Dispute rate: raw vs. corrected, Jan–Aug 2026")
    ax.set_xlabel("Month")
    ax.set_ylabel("Dispute rate (%)")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left")
    ax.tick_params(axis="x", rotation=30)

    last = trend.iloc[-1]
    ax.annotate(
        f"+{last['dispute_rate_raw'] * 100 - trend.iloc[-2]['dispute_rate_raw'] * 100:.2f}pp raw\n"
        f"+{last['dispute_rate_corrected'] * 100 - trend.iloc[-2]['dispute_rate_corrected'] * 100:.2f}pp corrected",
        xy=(trend["month"].iloc[-1], last["dispute_rate_raw"] * 100),
        xytext=(-140, -6),
        textcoords="offset points",
        fontsize=8.5,
        color=COLOR_NEUTRAL,
        arrowprops={"arrowstyle": "->", "color": COLOR_NEUTRAL, "lw": 0.8},
    )
    ax.margins(y=0.15)

    fig.tight_layout()
    out = FIGURES_DIR / "dispute_rate_trend.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_anomaly_detection() -> Path:
    tables = resolve_tables()
    series = monthly_dispute_rate_series(tables[TABLE_TRANSACTIONS]).sort_index()
    mean, std = rolling_baseline(series, DEFAULT_THRESHOLDS.window, DEFAULT_THRESHOLDS.min_periods)
    upper = mean + DEFAULT_THRESHOLDS.fail_z * std
    lower = mean - DEFAULT_THRESHOLDS.fail_z * std

    months = series.index.tolist()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    # Plot the full-width series FIRST: matplotlib assigns a categorical/string
    # x-axis's tick positions by first-appearance order across every plotting
    # call on the Axes. Calling fill_between() first, with only the subset of
    # months that have a scored baseline, would give those months positions
    # 0..4 and push the unscored early months to the end instead of the front.
    ax.plot(months, series.to_numpy() * 100, marker="o", color=COLOR_NEUTRAL, label="Raw dispute rate")
    scored = mean.notna()
    ax.fill_between(
        [m for m, ok in zip(months, scored) if ok],
        (lower[scored] * 100),
        (upper[scored] * 100),
        color=COLOR_CORRECTED,
        alpha=0.12,
        label=f"Expected range (±{DEFAULT_THRESHOLDS.fail_z:g}σ of rolling baseline)",
    )

    anomaly_month = series.index[-1]
    ax.scatter(
        [anomaly_month],
        [series.loc[anomaly_month] * 100],
        color=COLOR_ANOMALY,
        s=90,
        zorder=5,
        label="Flagged anomaly (2026-08)",
    )
    ax.annotate(
        "kpi_anomaly__dispute_rate: fail",
        xy=(anomaly_month, series.loc[anomaly_month] * 100),
        xytext=(-150, -30),
        textcoords="offset points",
        fontsize=8.5,
        color=COLOR_ANOMALY,
        arrowprops={"arrowstyle": "->", "color": COLOR_ANOMALY, "lw": 0.8},
    )

    ax.set_title("Data-quality check: dispute rate vs. its rolling baseline")
    ax.set_xlabel("Month")
    ax.set_ylabel("Dispute rate (%)")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    out = FIGURES_DIR / "dispute_rate_anomaly.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_top_segment_drivers(top_n: int = 8) -> Path:
    report = build_driver_report(top_n=top_n)
    drivers = report.top_count_drivers.copy()
    drivers = drivers.sort_values("contribution_share_of_positive_dispute_change", ascending=True)
    labels = [f"{row.segment_name} = {row.segment_value}" for row in drivers.itertuples()]
    shares = drivers["contribution_share_of_positive_dispute_change"] * 100

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    colors = [COLOR_ANOMALY if v == shares.max() else COLOR_CORRECTED for v in shares]
    ax.barh(labels, shares, color=colors)
    for y, value in enumerate(shares):
        ax.text(value + 1, y, f"{value:.1f}%", va="center", fontsize=8.5, color=COLOR_NEUTRAL)

    ax.set_title(f"Top segment drivers of the corrected dispute-rate increase ({report.current_period})")
    ax.set_xlabel("Share of the positive dispute change (%)")
    ax.set_xlim(0, max(shares.max() * 1.18, 10))
    ax.grid(True, axis="x")

    fig.tight_layout()
    out = FIGURES_DIR / "top_segment_drivers.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _find_chrome() -> str | None:
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome:
        return chrome
    mac_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    return str(mac_chrome) if mac_chrome.exists() else None


async def _capture_via_cdp(cdp_port: int, app_url: str, out: Path, render_wait_seconds: float) -> None:
    """Navigate, wait for Streamlit's async render (not just the page-load event), screenshot.

    A plain `chrome --screenshot=file URL` CLI capture fires right after the
    page-load event, which for a Streamlit app is just the static shell --
    the real content renders afterwards, over a websocket, once the Python
    script finishes running. Driving Chrome over the DevTools Protocol
    instead lets us wait past that.
    """
    import base64
    import json

    import requests
    import websockets

    tab = requests.put(f"http://localhost:{cdp_port}/json/new?{app_url}", timeout=10).json()
    ws_url = tab["webSocketDebuggerUrl"]

    async with websockets.connect(ws_url, max_size=None) as ws:
        async def call(method: str, params: dict | None = None) -> dict:
            msg_id = call.counter = getattr(call, "counter", 0) + 1
            await ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
            while True:
                reply = json.loads(await ws.recv())
                if reply.get("id") == msg_id:
                    return reply

        await call("Page.enable")
        await call("Page.navigate", {"url": app_url})
        await asyncio.sleep(render_wait_seconds)
        result = await call(
            "Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True}
        )
        png_bytes = base64.b64decode(result["result"]["data"])
        out.write_bytes(png_bytes)


def chart_app_screenshot(
    port: int = 8799, cdp_port: int = 9399, render_wait_seconds: float = 20.0
) -> Path | None:
    """Launch the app headless and screenshot it via Chrome's DevTools Protocol.

    Returns None (and prints why) rather than raising, so a missing browser
    doesn't fail the other three figures -- this one is the least essential
    and the easiest to regenerate manually with a real browser if needed.
    """
    chrome = _find_chrome()
    if chrome is None:
        print("Skipping app_screenshot.png: no Chrome/Chromium binary found.")
        return None

    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_bin = str(venv_python) if venv_python.exists() else "python"
    app_process = subprocess.Popen(
        [
            python_bin,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "app" / "streamlit_app.py"),
            "--server.headless",
            "true",
            "--server.port",
            str(port),
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    chrome_process = subprocess.Popen(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            "--window-size=1440,1400",
            f"--remote-debugging-port={cdp_port}",
            "--user-data-dir=" + str(PROJECT_ROOT / "outputs" / "_chrome_profile"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    out = FIGURES_DIR / "app_screenshot.png"
    try:
        time.sleep(3)  # let both processes finish starting up
        asyncio.run(
            _capture_via_cdp(cdp_port, f"http://localhost:{port}", out, render_wait_seconds)
        )
    except Exception as exc:  # noqa: BLE001 - any capture failure just skips this one figure
        print(f"Skipping app_screenshot.png: {exc}")
        return None
    finally:
        for proc in (chrome_process, app_process):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    if not out.exists() or out.stat().st_size == 0:
        print("Skipping app_screenshot.png: capture produced no image.")
        return None
    return out


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    written = [
        chart_dispute_rate_trend(),
        chart_anomaly_detection(),
        chart_top_segment_drivers(),
    ]
    screenshot = chart_app_screenshot()
    if screenshot is not None:
        written.append(screenshot)

    print(f"Wrote {len(written)} figure(s) to {FIGURES_DIR}:")
    for path in written:
        size_kb = path.stat().st_size / 1024
        print(f"  - {path.name} ({size_kb:.0f} KB)")
