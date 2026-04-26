import os
import json
import hashlib
import statistics
from collections import defaultdict

import gradio as gr

_orig_preprocess = gr.Slider.preprocess
def _safe_preprocess(self, payload):
    if isinstance(payload, (list, tuple)):
        return payload
    return _orig_preprocess(self, payload)
gr.Slider.preprocess = _safe_preprocess
import plotly.graph_objects as go
import numpy as np

from osr_parser import OsuReplay, GameMode, ModsBit
from analyze_offsets import (
    parse_beatmap, parse_replay_frames, extract_actions,
    action_driven_judge, build_statistics, compute_windows,
    build_output, build_judgment_stats, classify_judgment,
    classify_combined_hold,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_DIR = os.path.join(THIS_DIR, "cases")
SETTINGS_PATH = os.path.join(THIS_DIR, "settings.json")
TEMP_DIR = os.path.join(THIS_DIR, ".webui_tmp")
os.makedirs(TEMP_DIR, exist_ok=True)

JUDGMENT_ORDER = ["PERFECT", "GREAT", "GOOD", "OK", "MEH", "MISS"]
OSU_TO_ANALYZER = {
    "PERFECT (Geki)": "PERFECT",
    "GREAT (300)": "GREAT",
    "GOOD (200/Katu)": "GOOD",
    "OK (100)": "OK",
    "MEH (50)": "MEH",
    "MISS": "MISS",
}
JUDGMENT_COLORS = {
    "PERFECT": "#00cc00", "GREAT": "#66ff66", "GOOD": "#00ccff",
    "OK": "#ffcc00", "MEH": "#ff8800", "MISS": "#ff2200",
}
COLORS_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]
MODE_LABELS = ["press (tap+head)", "hold_head", "hold_tail", "all (tap+head+tail)"]


def load_settings():
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    default = {"songs_path": ""}
    save_settings(default)
    return default


def save_settings(cfg):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def compute_file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_osu_by_md5(target_md5):
    settings = load_settings()
    search_dirs = [CASES_DIR]
    if settings.get("songs_path"):
        search_dirs.append(settings["songs_path"])
    for root_dir in search_dirs:
        if not os.path.isdir(root_dir):
            continue
        for dirpath, _, fnames in os.walk(root_dir):
            for fn in fnames:
                if fn.lower().endswith(".osu"):
                    full = os.path.join(dirpath, fn)
                    try:
                        if compute_file_md5(full) == target_md5:
                            return full
                    except Exception:
                        continue
    return None


def parse_replay_filename(filename):
    """
    Parse osu! replay filename to extract artist, title, difficulty.
    Expected format: "player - artist - title [difficulty] (date) mode.osr"
    Example: "Tomor1n - MIMI feat. wanko - Minimum [Luminesense] (2025-11-22) OsuMania.osr"
    Returns dict with artist, title, difficulty or None.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split(" - ")
    if len(parts) < 3:
        return None
    artist = parts[1]
    rest = " - ".join(parts[2:])
    lb = rest.find("[")
    rb = rest.find("]")
    if lb == -1 or rb == -1 or lb == 0:
        return None
    title = rest[:lb].strip()
    difficulty = rest[lb + 1:rb].strip()
    return {"artist": artist, "title": title, "difficulty": difficulty}


def find_osu_by_filename(osr_path, songs_path):
    """
    Search Songs directory for a .osu file matching replay filename.
    .osu files are named: "Artist - Title (Mapper) [Difficulty].osu"
    Matches by title (fuzzy) + [difficulty] (exact bracket).
    """
    info = parse_replay_filename(os.path.basename(osr_path))
    if not info or not os.path.isdir(songs_path):
        return None
    diff_bracket = f"[{info['difficulty']}]"
    title_lower = info["title"].lower()
    for dirpath, _, fnames in os.walk(songs_path):
        for fn in fnames:
            if not fn.lower().endswith(".osu"):
                continue
            if diff_bracket not in fn:
                continue
            fn_name = os.path.splitext(fn)[0]
            if title_lower in fn_name.lower():
                return os.path.join(dirpath, fn)
    return None


def compute_press_durations(frames, cs):
    by_col = {c: [] for c in range(cs)}
    prev_x = frames[0]["key_state"] if frames else 0
    active = {c: None for c in range(cs)}
    for f in frames[1:]:
        x = f["key_state"]
        if x != prev_x:
            changed = prev_x ^ x
            for col in range(cs):
                if changed & (1 << col):
                    is_press = (x >> col) & 1
                    t = f["time"]
                    if is_press:
                        active[col] = t
                    else:
                        if active[col] is not None:
                            dur = t - active[col]
                            if 0 < dur < 5000:
                                by_col[col].append((active[col], dur))
                            active[col] = None
        prev_x = x
    return by_col


def build_histogram_figure(state, mode_label, selected_keys, bin_count=80):
    if not state:
        return go.Figure()
    type_map = {
        "press (tap+head)": ["tap", "hold_head"],
        "hold_head": ["hold_head"],
        "hold_tail": ["hold_tail"],
        "all (tap+head+tail)": ["tap", "hold_head", "hold_tail"],
    }
    types = type_map.get(mode_label, ["tap", "hold_head"])
    keys = [int(k) for k in selected_keys]

    all_vals = []
    col_data = {}
    for col in keys:
        vals = [o["offset_ms"] for o in state["all_offsets"]
                if o["column"] == col and o["type"] in types
                and o["offset_ms"] is not None]
        if vals:
            col_data[col] = vals
            all_vals.extend(vals)

    if not all_vals:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        fig.update_layout(title="Offset Distribution")
        return fig

    w = state["windows"]
    max_abs = max(max(all_vals), -min(all_vals), w["miss"] + 50)
    nbins = max(bin_count, 5)
    bin_width = 2 * max_abs / nbins

    fig = go.Figure()
    for idx, col in enumerate(sorted(col_data.keys())):
        vals = col_data[col]
        color = COLORS_PALETTE[idx % len(COLORS_PALETTE)]
        avg = statistics.mean(vals)
        sd = statistics.pstdev(vals)
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=nbins,
            name=f"K{col} (n={len(vals)}, μ={avg:.1f}, σ={sd:.1f})",
            marker_color=color, opacity=0.55,
            xbins=dict(start=-max_abs, end=max_abs, size=bin_width),
        ))
        fig.add_vline(x=avg, line=dict(color=color, width=1.5, dash="solid"), opacity=0.7)
        fig.add_vline(x=avg - sd, line=dict(color=color, width=1, dash="dot"), opacity=0.4)
        fig.add_vline(x=avg + sd, line=dict(color=color, width=1, dash="dot"), opacity=0.4)

    bounds = [
        (w["perfect"], "#00cc00", "PERFECT"),
        (w["great"], "#66ff66", "GREAT"),
        (w["good"], "#00ccff", "GOOD"),
        (w["ok"], "#ffcc00", "OK"),
        (w["meh"], "#ff8800", "MEH"),
        (w["miss"], "#ff2200", "MISS"),
    ]
    for v, c, label in bounds:
        fig.add_vline(x=v, line=dict(color=c, width=0.8, dash="dash"), opacity=0.4)
        fig.add_vline(x=-v, line=dict(color=c, width=0.8, dash="dash"), opacity=0.4)
    fig.add_vline(x=0, line=dict(color="black", width=1.5), opacity=0.8)

    # Legend entry for judgment windows (only once)
    for v, c, label in bounds:
        if v == w["perfect"]:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=c, width=0.8, dash="dash"),
                name=f"±{label} (±{v}ms)",
            ))

    meta = state.get("meta", {})
    bm = meta.get("beatmap", {})
    title_str = (f"Offset Distribution ({mode_label})<br>"
                 f"{meta.get('player','?')} — {bm.get('artist','?')} - "
                 f"{bm.get('title','?')} [{bm.get('version','?')}]")

    fig.update_layout(
        title=dict(text=title_str, font=dict(size=13)),
        xaxis_title="Offset (ms)", yaxis_title="Count",
        barmode="overlay", hovermode="x unified",
        legend=dict(font=dict(size=10), itemsizing="constant"),
        xaxis=dict(range=[-max_abs, max_abs]),
        margin=dict(t=80, r=60, b=60, l=60),
        height=450,
    )
    return fig


def build_press_time_figure(state):
    if not state:
        return go.Figure()
    pds = state.get("press_durations", {})
    if not pds or all(len(v) == 0 for v in pds.values()):
        fig = go.Figure()
        fig.add_annotation(text="No press duration data", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper")
        return fig

    fig = go.Figure()
    for idx, col in enumerate(sorted(pds.keys())):
        durs = [d for _, d in pds[col]]
        if not durs:
            continue
        color = COLORS_PALETTE[idx % len(COLORS_PALETTE)]
        avg = statistics.mean(durs)
        sd = statistics.pstdev(durs)
        hist, edges = np.histogram(durs, bins=120)
        centers = (edges[:-1] + edges[1:]) / 2
        fig.add_trace(go.Scatter(
            x=centers, y=hist, mode="lines",
            name=f"K{col} (n={len(durs)}, μ={avg:.1f}ms, σ={sd:.1f}ms)",
            line=dict(color=color, width=1.2),
        ))

    fig.update_layout(
        title="Press Duration Distribution by Key",
        xaxis_title="Press Duration (ms)", yaxis_title="Count",
        hovermode="x unified",
        legend=dict(font=dict(size=10)),
        margin=dict(t=40, r=40, b=50, l=50),
        height=400,
    )
    return fig


def build_rolling_charts(state, window_size_ms):
    if not state:
        return go.Figure(), go.Figure()
    offsets = state.get("all_offsets", [])
    valid = sorted(
        [(o["hit_time_ms"], o["offset_ms"])
         for o in offsets if o["offset_ms"] is not None],
        key=lambda x: x[0])
    if len(valid) < 2:
        return go.Figure(), go.Figure()

    times = [v[0] for v in valid]
    vals = [v[1] for v in valid]
    t_min, t_max = times[0], times[-1]
    step = max(int((t_max - t_min) / 200), 1)
    positions = list(range(int(t_min), int(t_max + 1), step))

    pos_times, means, urs = [], [], []
    for center in positions:
        cumulative = [v for t, v in zip(times, vals) if t <= center]
        if len(cumulative) >= 2:
            pos_times.append(center / 1000.0)
            means.append(statistics.mean(cumulative))
            urs.append(statistics.pstdev(cumulative) * 10)

    fig_avg = go.Figure()
    fig_avg.add_trace(go.Scatter(
        x=pos_times, y=means, mode="lines",
        name="Avg Offset", line=dict(width=1.5, color="#4363d8")))
    fig_avg.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_avg.update_layout(
        title="Average Offset (cumulative)",
        xaxis_title="Time (s)", yaxis_title="Offset (ms)",
        margin=dict(t=40, r=30, b=40, l=50), height=280,
        hovermode="x unified",
    )

    fig_ur = go.Figure()
    fig_ur.add_trace(go.Scatter(
        x=pos_times, y=urs, mode="lines",
        name="UR", line=dict(width=1.5, color="#ff6600")))
    fig_ur.update_layout(
        title="Unstable Rate (cumulative, std × 10)",
        xaxis_title="Time (s)", yaxis_title="UR",
        margin=dict(t=40, r=30, b=40, l=50), height=280,
        hovermode="x unified",
    )
    return fig_avg, fig_ur


PREVIEW_COLORS = {
    "PERFECT": "#ffffff", "GREAT": "#ffd700", "GOOD": "#00cc00",
    "OK": "#4488ff", "MEH": "#888888", "MISS": "#ff2200",
}
PREVIEW_COL_WIDTH = 50
PREVIEW_COL_GAP = 4
PREVIEW_AXIS_W = 55
PREVIEW_PAD = 10
PREVIEW_TOP = 20
PREVIEW_SCALE = 0.12


def filter_offsets_in_range(offsets, t_min, t_max):
    """Filter offsets to those whose hit_time is within [t_min, t_max]."""
    return [o for o in offsets
            if o["offset_ms"] is not None and t_min <= o["hit_time_ms"] <= t_max]


def build_preview_data(notes, offsets, windows, scorev2, t_min=None, t_max=None):
    """Build preview entries matching analyzer judgment counts."""
    # Filter notes by time range
    fnotes = notes
    if t_min is not None:
        fnotes = [n for n in fnotes if n["time"] >= t_min]
    if t_max is not None:
        fnotes = [n for n in fnotes if n["time"] <= t_max]

    offset_map = {}
    release_map = {}
    for o in offsets:
        if o["offset_ms"] is None:
            continue
        key = (o["hit_time_ms"], o["column"])
        offset_map.setdefault(key, []).append(o)
        if o["type"] == "hold_tail":
            release_map[(o["hit_time_ms"], o["column"])] = o["event_time_ms"]

    result = []
    for n in fnotes:
        key = (n["time"], n["column"])
        entries = offset_map.get(key, [])
        is_hold = n["type"] == "hold"
        end_t = n.get("end_time", 0) or 0

        if not entries:
            result.append({
                "column": n["column"], "hit_time": n["time"],
                "end_time": end_t, "is_hold": is_hold, "is_tail": False,
                "judgment": "MISS", "event_time": None, "release_time": None,
            })
            if is_hold and end_t and scorev2:
                rt = release_map.get((end_t, n["column"]))
                result.append({
                    "column": n["column"], "hit_time": end_t,
                    "end_time": 0, "is_hold": False, "is_tail": True,
                    "judgment": "MISS", "event_time": rt, "release_time": None,
                })
            continue

        if is_hold and not scorev2:
            combined = [e for e in entries if e["type"] == "hold"]
            if combined:
                c = combined[0]
                jdg = classify_combined_hold(c["head_offset"], c["tail_offset"], windows)
                rt = release_map.get((end_t, n["column"]))
                result.append({
                    "column": n["column"], "hit_time": n["time"],
                    "end_time": end_t, "is_hold": True, "is_tail": False,
                    "judgment": jdg, "event_time": entries[0].get("event_time_ms"),
                    "release_time": rt,
                })
                continue

        # ScoreV2: produce head + tail as separate entries
        for e in entries:
            if e["type"] in ("tap", "hold_head") and e["offset_ms"] is not None:
                jdg = classify_judgment(e["offset_ms"], windows)
                head_et = e["event_time_ms"]
                tail_release = None
                if is_hold and end_t:
                    te = next((x for x in offset_map.get((end_t, n["column"]), [])
                               if x["type"] == "hold_tail" and x["offset_ms"] is not None), None)
                    if te is not None:
                        tail_release = te["event_time_ms"]
                result.append({
                    "column": n["column"], "hit_time": n["time"],
                    "end_time": end_t, "is_hold": is_hold, "is_tail": False,
                    "judgment": jdg, "event_time": head_et,
                    "release_time": tail_release,
                })
                # Tail entry for stats (no note block)
                if is_hold and te is not None:
                    tail_win = {k: v * 1.5 for k, v in windows.items()}
                    tail_jdg = classify_judgment(te["offset_ms"], tail_win if scorev2 else windows)
                    result.append({
                        "column": n["column"], "hit_time": end_t,
                        "end_time": 0, "is_hold": False, "is_tail": True,
                        "judgment": tail_jdg,
                        "event_time": te["event_time_ms"],
                        "release_time": None,
                    })
                break
    return result


def render_beatmap_svg(notes, preview_data, cs, max_time_ms,
                       scale=0.12, reverse=False, t_min=0, t_max=None):
    col_w = PREVIEW_COL_WIDTH
    col_gap = PREVIEW_COL_GAP
    axis_w = PREVIEW_AXIS_W
    pad = PREVIEW_PAD
    top = PREVIEW_TOP
    bg = "#111"
    text_color = "#aaa"
    grid_color = "#333"

    use_range = t_max is not None and t_max > t_min
    svg_w = pad + axis_w + pad + cs * (col_w + col_gap) + pad
    if use_range:
        range_h = t_max - t_min
        svg_h = int(range_h * scale) + top + 40
    else:
        svg_h = int(max_time_ms * scale) + top + 40

    def time_to_y(t):
        if use_range:
            if reverse:
                return (range_h - (t - t_min)) * scale + top
            return (t - t_min) * scale + top
        if reverse:
            return (max_time_ms - t) * scale + top
        return t * scale + top

    def col_x(c):
        return pad + axis_w + pad + c * (col_w + col_gap)

    note_w = col_w - 8
    note_h = 24
    press_h = 6
    press_w = col_w * 0.52
    note_ox = (col_w - note_w) / 2
    press_ox = (col_w - press_w) / 2

    lines = []
    lines.append(f'<svg width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg" '
                 f'style="background:{bg};font-family:sans-serif;font-size:10px">')

    # Grid lines (every 1000ms)
    step_ms = 1000
    for t in range(step_ms, int(max_time_ms) + step_ms, step_ms):
        y = time_to_y(t)
        if y < top - 5 or y > svg_h: break
        lines.append(f'<line x1="0" y1="{y:.1f}" x2="{svg_w}" y2="{y:.1f}" '
                     f'stroke="{grid_color}" stroke-width="0.5"/>')
        lines.append(f'<text x="{pad}" y="{y - 2:.1f}" fill="{text_color}">{t // 1000}s</text>')

    # Column labels
    for c in range(cs):
        x = col_x(c) + col_w / 2
        lines.append(f'<text x="{x}" y="{top - 5}" text-anchor="middle" fill="{text_color}" '
                     f'font-size="11">K{c}</text>')

    # Draw notes and presses
    for pd in preview_data:
        c = pd["column"]
        ht = pd["hit_time"]
        et = pd["event_time"]
        is_hold = pd["is_hold"]
        is_tail = pd.get("is_tail", False)
        rt = pd.get("release_time")
        jdg = pd["judgment"]
        color = PREVIEW_COLORS.get(jdg, "#888")

        nx = col_x(c) + note_ox
        px = col_x(c) + press_ox

        # Skip LN body and note block for tail entries
        if not is_tail:
            # LN body: full-width rect from head to tail
            if is_hold and pd.get("end_time"):
                y_s = time_to_y(ht)
                y_e = time_to_y(pd["end_time"])
                y_min = min(y_s, y_e)
                body_h = max(abs(y_e - y_s), 1)
                lines.append(f'<rect x="{nx:.1f}" y="{y_min:.1f}" width="{note_w}" '
                             f'height="{body_h:.1f}" fill="{color}" opacity="0.20" rx="1"/>')

            # Note block: filled rect with border
            y_note = time_to_y(ht) - note_h / 2
            lines.append(f'<rect x="{nx:.1f}" y="{y_note:.1f}" width="{note_w}" '
                         f'height="{note_h}" fill="{color}" opacity="0.25" rx="2"/>')
            lines.append(f'<rect x="{nx:.1f}" y="{y_note:.1f}" width="{note_w}" '
                         f'height="{note_h}" fill="none" stroke="{color}" '
                         f'stroke-width="2" rx="2"/>')

        # Press overlay at event_time (for both head and tail)
        if et is not None:
            y_press = time_to_y(et) - press_h / 2
            lines.append(f'<rect x="{px:.1f}" y="{y_press:.1f}" width="{press_w}" '
                         f'height="{press_h}" fill="{color}" rx="1"/>')

        # Press hold bar: from event_time to release_time (or end_time as fallback)
        if et is not None:
            hold_end = rt if rt else pd.get("end_time")
            if hold_end and hold_end > et:
                y_ps = time_to_y(et)
                y_pe = time_to_y(hold_end)
                y_pmin = min(y_ps, y_pe)
                p_h = max(abs(y_pe - y_ps), 1)
                lines.append(f'<rect x="{px:.1f}" y="{y_pmin:.1f}" width="{press_w}" '
                             f'height="{p_h:.1f}" fill="{color}" rx="1" opacity="0.30"/>')

    lines.append('</svg>')
    return "".join(lines)


def build_range_rolling_charts(filtered_offsets, window_ms=2000):
    """Cumulative avg + UR line charts for a filtered offset set."""
    valid = sorted(
        [(o["hit_time_ms"], o["offset_ms"])
         for o in filtered_offsets if o["offset_ms"] is not None],
        key=lambda x: x[0])
    empty = (go.Figure(), go.Figure())

    if len(valid) < 2:
        return empty

    times = [v[0] for v in valid]
    vals = [v[1] for v in valid]
    t_first, t_last = times[0], times[-1]
    step = max(int((t_last - t_first) / 200), 1)
    positions = list(range(int(t_first), int(t_last + 1), step))

    pos_t, means, urs = [], [], []
    for center in positions:
        cumulative = [v for t, v in zip(times, vals) if t <= center]
        if len(cumulative) >= 2:
            pos_t.append(center / 1000.0)
            means.append(statistics.mean(cumulative))
            urs.append(statistics.pstdev(cumulative) * 10)

    fig_avg = go.Figure()
    fig_avg.add_trace(go.Scatter(x=pos_t, y=means, mode="lines",
        name="Avg Offset", line=dict(width=1.5, color="#4363d8")))
    fig_avg.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_avg.update_layout(title="区间 Avg Offset (cumulative)", xaxis_title="Time (s)",
        yaxis_title="Offset (ms)", margin=dict(t=30, r=20, b=20, l=40), height=180, hovermode="x unified")

    fig_ur = go.Figure()
    fig_ur.add_trace(go.Scatter(x=pos_t, y=urs, mode="lines",
        name="UR", line=dict(width=1.5, color="#ff6600")))
    fig_ur.update_layout(title="区间 UR (cumulative)", xaxis_title="Time (s)",
        yaxis_title="UR", margin=dict(t=30, r=20, b=20, l=40), height=180, hovermode="x unified")

    return fig_avg, fig_ur


def build_range_judgment_table(filtered_offsets, windows, scorev2):
    from collections import Counter
    cnt = Counter()
    if scorev2:
        tail_win = {k: v * 1.5 for k, v in windows.items()}
    for o in filtered_offsets:
        if o["offset_ms"] is None:
            cnt["MISS"] += 1
            continue
        if scorev2 and o["type"] == "hold_tail":
            cnt[classify_judgment(o["offset_ms"], tail_win)] += 1
        elif not scorev2 and o["type"] == "hold":
            cnt[classify_combined_hold(o["head_offset"], o["tail_offset"], windows)] += 1
        elif o["type"] in ("tap", "hold_head", "hold_tail"):
            cnt[classify_judgment(o["offset_ms"], windows)] += 1
        else:
            cnt["MISS"] += 1
    headers = ["Total"] + JUDGMENT_ORDER
    row = [sum(cnt.values())] + [cnt.get(j, 0) for j in JUDGMENT_ORDER]
    return [row], headers


def update_range_view(state, t_range=None, zoom=None, reverse=None):
    if t_range is None:
        t_range = state.get("current_range", [0, state.get("max_time", 1000)])
    if zoom is None:
        zoom = 0.12
    if reverse is None:
        reverse = True
    t_min, t_max = t_range
    empty_html = '<div style="height:520px;border:1px solid #333;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#666;background:#111">请先分析一个回放</div>'
    empty_fig = go.Figure()
    empty_table = [["-"] * 7], ["Total"] + JUDGMENT_ORDER

    if not state or not state.get("beatmap_notes"):
        return [empty_html] + [empty_fig] * 4 + [gr.Dataframe(value=empty_table[0], headers=empty_table[1])]

    notes = state["beatmap_notes"]
    cs = state["meta"]["cs"]
    windows = state["windows"]
    scorev2 = state["scorev2"]
    offsets = state["all_offsets"]
    max_t = state.get("max_time", max((n.get("end_time",0) or 0) for n in notes) if notes else 0)

    # Filtered preview
    pd = build_preview_data(notes, offsets, windows, scorev2, t_min=t_min, t_max=t_max)
    svg = render_beatmap_svg(notes, pd, cs, max_t, scale=zoom, reverse=reverse,
                             t_min=t_min, t_max=t_max)
    preview_html = f'<div style="height:520px;overflow-y:auto;border:1px solid #333;border-radius:6px;background:#111">{svg}</div>'

    # Filtered offsets for range stats
    fo = filter_offsets_in_range(offsets, t_min, t_max)

    # Range avg + UR curves
    fig_avg, fig_ur = build_range_rolling_charts(fo, 2000)

    # Range press time (filtered from state's press_durations by press time)
    pds = state.get("press_durations", {})
    fig_pt = go.Figure()
    col_pds = {}
    for col, entries in pds.items():
        filtered = [d for pt, d in entries if t_min <= pt <= t_max]
        if filtered:
            col_pds[col] = filtered
    for idx, (col, durs) in enumerate(sorted(col_pds.items())):
        if not durs: continue
        c = COLORS_PALETTE[idx % len(COLORS_PALETTE)]
        avg_d = statistics.mean(durs)
        sd_d = statistics.pstdev(durs)
        hist, edges = np.histogram(durs, bins=60)
        centers = (edges[:-1] + edges[1:]) / 2
        fig_pt.add_trace(go.Scatter(x=centers, y=hist, mode="lines",
            name=f"K{col} (n={len(durs)}, μ={avg_d:.1f}, σ={sd_d:.1f})",
            line=dict(width=1.2, color=c)))
    fig_pt.update_layout(title="Press Duration (range)", xaxis_title="Press Duration (ms)", yaxis_title="Count",
                         height=200, margin=dict(t=30, b=20, l=30, r=10), hovermode="x unified")

    # Range histogram
    fig_hist = go.Figure()
    vals = [o["offset_ms"] for o in fo if o["offset_ms"] is not None]
    if vals:
        w = windows
        max_abs = max(max(vals), -min(vals), w["miss"] + 50)
        nbins = 60
        for col in sorted(set(o["column"] for o in fo)):
            cvals = [o["offset_ms"] for o in fo if o["column"] == col and o["offset_ms"] is not None]
            if not cvals: continue
            color = COLORS_PALETTE[col % len(COLORS_PALETTE)]
            fig_hist.add_trace(go.Histogram(x=cvals, nbinsx=nbins,
                name=f"K{col} (n={len(cvals)})", marker_color=color, opacity=0.6,
                xbins=dict(start=-max_abs, end=max_abs, size=2*max_abs/nbins)))
        for v, c, label in [(w["perfect"],"#00cc00","P"),(w["great"],"#66ff66","G"),
                            (w["good"],"#00ccff","Gd"),(w["ok"],"#ffcc00","O"),
                            (w["meh"],"#ff8800","Mh"),(w["miss"],"#ff2200","Ms")]:
            fig_hist.add_vline(x=v, line=dict(color=c, width=0.6, dash="dash"), opacity=0.3)
            fig_hist.add_vline(x=-v, line=dict(color=c, width=0.6, dash="dash"), opacity=0.3)
        fig_hist.update_layout(barmode="overlay", title="Offset (range)", height=200,
                               margin=dict(t=30, b=20, l=30, r=10), hovermode="x unified",
                               xaxis=dict(range=[-max_abs, max_abs]))

    # Range judgment table
    jdg_data, jdg_headers = build_range_judgment_table(fo, windows, scorev2)
    jdg_table = gr.Dataframe(value=jdg_data, headers=jdg_headers, datatype=["number"]*7, col_count=7, row_count=1)

    return [preview_html, fig_avg, fig_ur, fig_pt, fig_hist, jdg_table]


def run_analysis(osr_file_obj, osu_file_obj, songs_path_text):
    NONE20 = [None] * 20

    if osr_file_obj is None:
        return NONE20[:1] + [gr.Dataframe(value=None)] * 5 + NONE20[6:]

    osr_bytes = osr_file_obj if isinstance(osr_file_obj, bytes) else open(osr_file_obj, "rb").read()
    osr_path = os.path.join(TEMP_DIR, "upload.osr")
    with open(osr_path, "wb") as f:
        f.write(osr_bytes)

    try:
        replay = OsuReplay.from_file(osr_path)
    except Exception as e:
        return [gr.Markdown(f"⛔ .osr 解析失败: {e}")] + [gr.Dataframe(value=None)] * 5 + NONE20[6:18] + [gr.Slider(value=0), gr.Slider(value=0)]
    if replay.game_mode != GameMode.MANIA:
        return [gr.Markdown("⛔ 仅支持 osu!mania 模式 (game_mode=3)")] + [gr.Dataframe(value=None)] * 5 + NONE20[6:18] + [gr.Slider(value=0), gr.Slider(value=0)]

    beatmap_md5 = replay.beatmap_md5
    player = replay.player_name
    mods_names = ModsBit.names(replay.mods)

    osu_path = None
    if osu_file_obj is not None:
        osu_bytes = osu_file_obj if isinstance(osu_file_obj, bytes) else open(osu_file_obj, "rb").read()
        osu_path = os.path.join(TEMP_DIR, "upload.osu")
        with open(osu_path, "wb") as f:
            f.write(osu_bytes)

    if osu_path is None or not os.path.exists(osu_path):
        settings = load_settings()
        songs_path = settings.get("songs_path", "")
        osu_path = find_osu_by_filename(osr_path, songs_path)
    if osu_path is None or not os.path.exists(osu_path):
        osu_path = find_osu_by_md5(beatmap_md5)

    if osu_path is None or not os.path.exists(osu_path):
        replay_name = os.path.basename(osr_path)
        parsed = parse_replay_filename(replay_name)
        extra = ""
        if parsed:
            extra = (f"\n文件名解析成功: **{parsed['artist']}** — "
                     f"**{parsed['title']}** [{parsed['difficulty']}]"
                     f"\n但在 Songs 目录未找到匹配的 .osu 文件")
        return [gr.Markdown(
            f"✅ 回放已读取 (玩家: {player}, 模组: {mods_names})\n\n"
            f"⛔ 未找到对应谱面 (MD5: {beatmap_md5}){extra}\n"
            f"请上传对应的 .osu 文件或检查 Songs 路径配置")] + [gr.Dataframe(value=None)] * 5 + NONE20[6:]

    try:
        notes, cs, od, meta = parse_beatmap(osu_path)
    except Exception as e:
        return [gr.Markdown(f"⛔ .osu 解析失败: {e}")] + [gr.Dataframe(value=None)] * 5 + NONE20[6:]

    try:
        player2, mods_v, mods_n, frames = parse_replay_frames(osr_path)
    except Exception as e:
        return [gr.Markdown(f"⛔ Replay 帧解析失败: {e}")] + [gr.Dataframe(value=None)] * 5 + NONE20[6:]

    scorev2 = bool(replay.mods & ModsBit.SCORE_V2)
    windows = compute_windows(od, scorev2=scorev2)
    sv2_windows = compute_windows(od, scorev2=True)

    actions = extract_actions(frames, cs)
    offsets, missed, extra = action_driven_judge(notes, actions, windows, cs, scorev2=scorev2)
    match_extras = {"extra_presses": extra, "extra_releases": 0}
    overall_stats, by_type_stats, by_column_stats = build_statistics(offsets, windows)
    result = build_output(
        meta, player, replay.mods, mods_names, cs, od, windows,
        notes, offsets, match_extras, overall_stats, by_type_stats, by_column_stats,
    )

    # Build judgment stats matching osu! metadata semantics
    jdg_windows = sv2_windows if scorev2 else windows
    jdg_stats = build_judgment_stats(notes, offsets, jdg_windows, scorev2=scorev2)

    te = result["match_summary"]["total_entries"]
    me = result["match_summary"]["matched_entries"]
    ep = result["match_summary"]["extra_presses"]
    er = result["match_summary"]["extra_releases"]

    summary_data = [[result["match_summary"]["total_notes"], me, te - me, ep, er]]

    detail_headers = ["Column", "Type", "count", "mean(ms)", "median(ms)",
                      "std(ms)", "min(ms)", "max(ms)"] + JUDGMENT_ORDER
    detail_rows = []
    for col_data in by_column_stats:
        col = col_data["column"]
        for typ in ["tap", "hold_head", "hold_tail"]:
            if typ not in col_data:
                continue
            d = col_data[typ]
            jc = d.get("judgment_counts", {})
            row = [f"K{col}", typ, d["count"], d["mean"], d["median"],
                   d["std"], d["min"], d["max"]]
            row += [jc.get(j, 0) for j in JUDGMENT_ORDER]
            detail_rows.append(row)

    overall_rows = []
    for typ in ["tap", "hold_head", "hold_tail"]:
        if typ not in overall_stats:
            continue
        d = overall_stats[typ]
        jc = d.get("judgment_counts", {})
        row = ["Overall", typ, d["count"], d["mean"], d["median"],
               d["std"], d["min"], d["max"]]
        row += [jc.get(j, 0) for j in JUDGMENT_ORDER]
        overall_rows.append(row)

    # Merged summary row using judgment-stats (metadata-compatible)
    jdg_total = sum(jdg_stats.values())
    summary_row = ["**总计**",
                   "tap+hold_head+hold_tail" if scorev2 else "tap+hold",
                   jdg_total, "-", "-", "-", "-", "-"]
    summary_row += [jdg_stats.get(j, 0) for j in JUDGMENT_ORDER]

    title = result["metadata"]["beatmap"]["title"]
    artist = result["metadata"]["beatmap"]["artist"]
    version = result["metadata"]["beatmap"]["version"]
    status_text = (
        f"✅ **{player}** — {artist} - {title} [{version}]  "
        f"(CS={cs}, OD={od})  Mods: {mods_names if mods_names else 'None'}  |  "
        f"Matched: {me}/{te}  |  Extra: +{ep}p / +{er}r"
    )

    # Judgment comparison: .osr metadata vs analyzer
    osu_jdg = {
        "PERFECT (Geki)": replay.count_geki,
        "GREAT (300)": replay.count_300,
        "GOOD (200/Katu)": replay.count_katu,
        "OK (100)": replay.count_100,
        "MEH (50)": replay.count_50,
        "MISS": replay.count_miss,
    }
    comp_rows = []
    for osu_label, ana_label in OSU_TO_ANALYZER.items():
        osu_val = osu_jdg.get(osu_label, 0)
        ana_val = jdg_stats.get(ana_label, 0)
        diff = ana_val - osu_val
        comp_rows.append([osu_label, osu_val, ana_val, f"{diff:+d}" if diff != 0 else "0"])

    comp_headers = ["判定", "osu! 回放 (.osr)", "分析器结果", "差异"]

    press_durations = compute_press_durations(frames, cs)

    sorted_cols = sorted(set(o["column"] for o in result["offsets"]))
    max_time = int(max(
        max((n.get("end_time", 0) or 0) for n in notes),
        max(n["time"] for n in notes) if notes else 0,
    ))
    state = {
        "meta": {"player": player, "beatmap": {"artist": artist, "title": title, "version": version},
                 "cs": cs, "od": od},
        "windows": windows,
        "overall_stats": overall_stats,
        "by_column_stats": by_column_stats,
        "all_offsets": result["offsets"],
        "sorted_cols": sorted_cols,
        "press_durations": dict(press_durations),
        "osu_judgments": osu_jdg,
        "scorev2": scorev2,
        "beatmap_notes": notes,
        "max_time": max_time,
        "current_range": [0, max_time],
    }

    col_choices = [str(c) for c in sorted_cols]

    # Build beatmap preview
    preview_data = build_preview_data(notes, offsets, windows, scorev2)
    preview_scale = 0.12
    preview_svg = render_beatmap_svg(notes, preview_data, cs, max_time,
                                     scale=preview_scale, reverse=False)
    preview_html = f'<div style="height:520px;overflow-y:auto;border:1px solid #333;border-radius:6px;background:#111">{preview_svg}</div>'

    return [
        gr.Markdown(status_text),
        gr.Dataframe(value=summary_data,
                      headers=["总 Notes", "匹配数", "未匹配", "额外按压", "额外释放"]),
        gr.Dataframe(value=detail_rows, headers=detail_headers,
                     datatype=["str"] + ["number"] * 7 + ["number"] * 6),
        gr.Dataframe(value=overall_rows, headers=detail_headers,
                     datatype=["str"] + ["number"] * 7 + ["number"] * 6),
        gr.Dataframe(value=[summary_row], headers=detail_headers,
                     datatype=["str"] + ["number"] * 7 + ["number"] * 6),
        gr.Dataframe(value=comp_rows, headers=comp_headers,
                     datatype=["str"] + ["number"] * 2 + ["str"]),
        state,
        gr.Radio(value="press (tap+head)", choices=MODE_LABELS),
        gr.CheckboxGroup(value=col_choices, choices=col_choices, label="显示列", interactive=True),
        build_histogram_figure(state, "press (tap+head)", col_choices, 80),
        build_press_time_figure(state),
        gr.Slider(minimum=500, maximum=10000, value=2000, step=100,
                  label="滑动窗口大小 (ms)"),
        gr.Slider(minimum=10, maximum=200, value=80, step=5,
                  label="直方图 bin 数量"),
        *build_rolling_charts(state, 2000),
        gr.HTML(value=preview_html),
        gr.Slider(minimum=0.1, maximum=0.80, value=0.25, step=0.01,
                  label="缩放 (像素/ms)"),
        gr.Checkbox(value=True, label="反向 (时间向上)"),
        gr.Slider(minimum=0, maximum=max_time, value=0, step=100,
                  label="区间起点 (ms)"),
        gr.Slider(minimum=0, maximum=max_time, value=max_time, step=100,
                  label="区间终点 (ms)"),
    ]


def save_songs_path(path):
    cfg = load_settings()
    cfg["songs_path"] = path
    save_settings(cfg)
    exists = os.path.isdir(path)
    return gr.Markdown(f"已保存: {path}\n{'✅ 目录存在' if exists else '⛔ 目录不存在'}")


def build_ui():
    settings = load_settings()
    songs_path = settings.get("songs_path", "")

    with gr.Blocks(title="osu!mania Replay Offset Analyzer",
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎮 osu!mania Replay Offset Analyzer")

        with gr.Row():
            osr_input = gr.File(label="上传 .osr (回放)", file_types=[".osr"])
            osu_input = gr.File(label="上传 .osu (谱面，可选)", file_types=[".osu"])

        with gr.Row():
            songs_path_input = gr.Textbox(label="osu! Songs 目录 (自动搜 .osu)",
                                          value=songs_path,
                                          placeholder="例如: D:\\osumap\\osu!\\Songs\\",
                                          scale=4)
            save_path_btn = gr.Button("保存路径", scale=1)

        with gr.Row():
            analyze_btn = gr.Button("📊 Analyze", variant="primary", scale=1)

        save_path_msg = gr.Markdown("")
        status = gr.Markdown("等待文件...")

        with gr.Row():
            summary_table = gr.Dataframe(label="匹配总览", col_count=5,
                                         datatype=["number"] * 5, row_count=1)

        # --- Foldable statistics ---
        merged_table = gr.Dataframe(label="判定总计 (tap+hold_head+hold_tail 合并)",
                                    datatype=["str"] + ["number"] * 7 + ["number"] * 6,
                                    col_count=14, row_count=1)
        with gr.Accordion("📊 逐列 / 逐类型 统计详情", open=False):
            with gr.Row():
                detail_table = gr.Dataframe(label="逐列 / 逐类型统计",
                                            datatype=["str"] + ["number"] * 7 + ["number"] * 6,
                                            col_count=14)
                overall_table = gr.Dataframe(label="整体统计",
                                             datatype=["str"] + ["number"] * 7 + ["number"] * 6,
                                             col_count=14)

        # --- Judgment comparison ---
        comparison_table = gr.Dataframe(label="判定对比: osu! 回放 vs 分析器",
                                        datatype=["str"] + ["number"] * 2 + ["str"],
                                        col_count=4)

        state = gr.State()

        with gr.Row():
            mode_radio = gr.Radio(choices=MODE_LABELS, value="press (tap+head)",
                                  label="显示模式", interactive=True)
            col_checkbox = gr.CheckboxGroup(choices=[], value=[], label="显示列",
                                            interactive=True)
            bin_slider = gr.Slider(minimum=10, maximum=200, value=80, step=5,
                                   label="直方图 bin 数", scale=1)

        gr.HTML("""
<style>
.fs-wrap { position: relative; }
.fs-btn {
    position: absolute; top: 4px; right: 8px; z-index: 100;
    background: rgba(128,128,128,0.15); border: none; border-radius: 4px;
    cursor: pointer; font-size: 18px; line-height: 1; padding: 2px 6px;
    color: inherit; transition: background 0.15s;
}
.fs-btn:hover { background: rgba(128,128,128,0.4); }
.fs-wrap:-webkit-full-screen { background: var(--bg,#fff); padding: 20px; }
.fs-wrap:-moz-full-screen { background: var(--bg,#fff); padding: 20px; }
.fs-wrap:fullscreen { background: var(--bg,#fff); padding: 20px; }
.fs-wrap:-webkit-full-screen .plotly-graph-div { width: 100% !important; height: 100% !important; }
.fs-wrap:fullscreen .plotly-graph-div { width: 100% !important; height: 100% !important; }
</style>
""")

        with gr.Row():
            with gr.Column(scale=1, elem_classes="fs-wrap"):
                dist_plot = gr.Plot(label="偏移分布直方图", elem_id="hist-plot")
                gr.HTML('<button class="fs-btn" onclick="document.getElementById(\'hist-plot\').closest(\'.fs-wrap\').requestFullscreen()">⛶</button>')
            with gr.Column(scale=1, elem_classes="fs-wrap"):
                press_time_plot = gr.Plot(label="按压时长分布 (PressTime)", elem_id="press-plot")
                gr.HTML('<button class="fs-btn" onclick="document.getElementById(\'press-plot\').closest(\'.fs-wrap\').requestFullscreen()">⛶</button>')

        # --- Rolling charts (replaces pie) ---
        window_slider = gr.Slider(
            minimum=500, maximum=10000, value=2000, step=100,
            label="滑动窗口大小 (ms)")
        rolling_avg_plot = gr.Plot(label="平均偏移随时间变化")
        ur_plot = gr.Plot(label="UR 随时间变化 (std × 10)")

        # --- Beatmap preview with range selector ---
        with gr.Row():
            range_min = gr.Slider(minimum=0, maximum=600000, value=0, step=100,
                                  label="区间起点 (ms)", scale=1)
            range_max = gr.Slider(minimum=0, maximum=600000, value=60000, step=100,
                                  label="区间终点 (ms)", scale=1)
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    zoom_slider = gr.Slider(minimum=0.04, maximum=0.40, value=0.20, step=0.01,
                                            label="缩放 (像素/ms)", scale=3)
                    direction_checkbox = gr.Checkbox(value=True, label="反向 (时间向上)", scale=1)
                preview_html = gr.HTML(value="<div style='height:520px;border:1px solid #333;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#666;background:#111'>请先分析一个回放</div>")

            with gr.Column(scale=1):
                range_avg_plot = gr.Plot(label="区间 Avg Offset")
                range_ur_plot = gr.Plot(label="区间 UR")
                range_presstime_plot = gr.Plot(label="区间 PressTime")
                range_histogram_plot = gr.Plot(label="区间 Histogram")
                range_judgment_table = gr.Dataframe(label="区间判定分布", col_count=7,
                                                    datatype=["number"] * 7, row_count=1)

        # Events
        EMPTY_16 = [None] * 16
        analyze_btn.click(
            fn=run_analysis,
            inputs=[osr_input, osu_input, songs_path_input],
    outputs=[status, summary_table, detail_table, overall_table,
             merged_table, comparison_table,
             state, mode_radio, col_checkbox,
             dist_plot, press_time_plot,
             window_slider, bin_slider, rolling_avg_plot, ur_plot,
             preview_html, zoom_slider, direction_checkbox, range_min, range_max],
        )

        for trigger in [mode_radio, col_checkbox, bin_slider]:
            trigger.change(
                fn=build_histogram_figure,
                inputs=[state, mode_radio, col_checkbox, bin_slider],
                outputs=dist_plot,
            )

        window_slider.change(
            fn=build_rolling_charts,
            inputs=[state, window_slider],
            outputs=[rolling_avg_plot, ur_plot],
        )

        range_outputs = [preview_html, range_avg_plot, range_ur_plot,
                         range_presstime_plot, range_histogram_plot, range_judgment_table]

        # Range view triggers
        def on_range_or_zoom_change(state, rmin, rmax, zoom, rev):
            t_range = [rmin, rmax]
            state["current_range"] = t_range
            return update_range_view(state, t_range, zoom, rev)

        for trigger in [range_min, range_max, zoom_slider, direction_checkbox]:
            trigger.change(
                fn=on_range_or_zoom_change,
                inputs=[state, range_min, range_max, zoom_slider, direction_checkbox],
                outputs=range_outputs,
            )

        save_path_btn.click(fn=save_songs_path, inputs=songs_path_input,
                            outputs=save_path_msg)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860)
