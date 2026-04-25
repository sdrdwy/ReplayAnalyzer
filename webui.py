import os
import json
import hashlib
import statistics
from collections import defaultdict

import gradio as gr
import plotly.graph_objects as go
import numpy as np

from osr_parser import OsuReplay, GameMode, ModsBit
from analyze_offsets import (
    parse_beatmap, parse_replay_frames, detect_events,
    match_events, build_statistics, compute_windows,
    build_output,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_DIR = os.path.join(THIS_DIR, "cases")
CONFIG_PATH = os.path.join(THIS_DIR, "config.json")
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


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"songs_path": ""}


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def compute_file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_osu_by_md5(target_md5):
    config = load_config()
    search_dirs = [CASES_DIR]
    if config.get("songs_path"):
        search_dirs.append(config["songs_path"])
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
                                by_col[col].append(dur)
                            active[col] = None
        prev_x = x
    return by_col


def build_histogram_figure(state, mode_label, selected_keys):
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
    nbins = 80
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
        durs = pds[col]
        if not durs:
            continue
        color = COLORS_PALETTE[idx % len(COLORS_PALETTE)]
        avg = statistics.mean(durs)
        sd = statistics.pstdev(durs)
        hist, edges = np.histogram(durs, bins=40)
        centers = (edges[:-1] + edges[1:]) / 2
        fig.add_trace(go.Scatter(
            x=centers, y=hist, mode="lines+markers",
            name=f"K{col} (n={len(durs)}, μ={avg:.1f}ms, σ={sd:.1f}ms)",
            line=dict(color=color), marker=dict(color=color, size=4),
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


def build_pie_figure(overall_stats):
    if not overall_stats:
        return go.Figure()
    fig = go.Figure()
    for idx, typ in enumerate(["tap", "hold_head", "hold_tail"]):
        if typ not in overall_stats:
            continue
        jdg_counts = overall_stats[typ].get("judgment_counts", {})
        labels = [j for j in JUDGMENT_ORDER if jdg_counts.get(j, 0) > 0]
        values = [jdg_counts.get(j, 0) for j in labels]
        colors = [JUDGMENT_COLORS.get(j, "#888") for j in labels]
        fig.add_trace(go.Pie(
            labels=labels, values=values,
            name=typ, domain=dict(row=0, column=idx),
            marker_colors=colors, hole=0.3,
            textinfo="label+percent", textfont_size=10,
        ))
    title_map = {"tap": "Tap", "hold_head": "Hold Head", "hold_tail": "Hold Tail"}
    annotations = []
    for idx, typ in enumerate(["tap", "hold_head", "hold_tail"]):
        if typ in overall_stats:
            annotations.append(dict(
                text=f"{title_map[typ]} (n={overall_stats[typ]['count']})",
                x=idx / 2, y=1.05, showarrow=False, font=dict(size=11),
                xref="paper", yref="paper"))
    fig.update_layout(
        title="Judgment Distribution by Type",
        grid=dict(rows=1, columns=3),
        annotations=annotations,
        height=350,
        margin=dict(t=50, b=30),
    )
    return fig


def run_analysis(osr_file_obj, osu_file_obj, songs_path_text):
    if osr_file_obj is None:
        return [gr.Markdown("⛔ 请上传 .osr 文件")] + [gr.Dataframe(value=None)] * 5 + [None, None, None, None, None, None]

    osr_bytes = osr_file_obj if isinstance(osr_file_obj, bytes) else open(osr_file_obj, "rb").read()
    osr_path = os.path.join(TEMP_DIR, "upload.osr")
    with open(osr_path, "wb") as f:
        f.write(osr_bytes)

    try:
        replay = OsuReplay.from_file(osr_path)
    except Exception as e:
        return [gr.Markdown(f"⛔ .osr 解析失败: {e}")] + [gr.Dataframe(value=None)] * 10

    if replay.game_mode != GameMode.MANIA:
        return [gr.Markdown("⛔ 仅支持 osu!mania 模式 (game_mode=3)")] + [gr.Dataframe(value=None)] * 10

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
        osu_path = find_osu_by_md5(beatmap_md5)

    if osu_path is None or not os.path.exists(osu_path):
        return [gr.Markdown(
            f"✅ 回放已读取 (玩家: {player}, 模组: {mods_names})\n\n"
            f"⛔ 未找到对应谱面 (MD5: {beatmap_md5})\n"
            f"请上传 .osu 或配置 Songs 路径")] + [gr.Dataframe(value=None)] * 10

    try:
        notes, cs, od, meta = parse_beatmap(osu_path)
    except Exception as e:
        return [gr.Markdown(f"⛔ .osu 解析失败: {e}")] + [gr.Dataframe(value=None)] * 10

    try:
        player2, mods_v, mods_n, frames = parse_replay_frames(osr_path)
    except Exception as e:
        return [gr.Markdown(f"⛔ Replay 帧解析失败: {e}")] + [gr.Dataframe(value=None)] * 10

    windows = compute_windows(od)
    press_events, release_events = detect_events(frames, cs)
    offsets, match_extras = match_events(notes, press_events, release_events, cs, windows["miss"])
    overall_stats, by_type_stats, by_column_stats = build_statistics(offsets, windows)
    result = build_output(
        meta, player, replay.mods, mods_names, cs, od, windows,
        notes, offsets, match_extras, overall_stats, by_type_stats, by_column_stats,
    )

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
    merged_jdg = defaultdict(int)
    for typ in ["tap", "hold_head", "hold_tail"]:
        if typ not in overall_stats:
            continue
        d = overall_stats[typ]
        jc = d.get("judgment_counts", {})
        row = ["Overall", typ, d["count"], d["mean"], d["median"],
               d["std"], d["min"], d["max"]]
        row += [jc.get(j, 0) for j in JUDGMENT_ORDER]
        overall_rows.append(row)
        for j in JUDGMENT_ORDER:
            merged_jdg[j] += jc.get(j, 0)

    merged_total = sum(merged_jdg.values())
    summary_row = ["**总计**", "tap+head+tail", merged_total, "-", "-", "-", "-", "-"]
    summary_row += [merged_jdg.get(j, 0) for j in JUDGMENT_ORDER]

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
        ana_val = merged_jdg.get(ana_label, 0)
        diff = ana_val - osu_val
        comp_rows.append([osu_label, osu_val, ana_val, f"{diff:+d}" if diff != 0 else "0"])

    comp_headers = ["判定", "osu! 回放 (.osr)", "分析器结果", "差异"]

    press_durations = compute_press_durations(frames, cs)

    sorted_cols = sorted(set(o["column"] for o in result["offsets"]))
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
    }

    col_choices = [str(c) for c in sorted_cols]

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
        build_histogram_figure(state, "press (tap+head)", col_choices),
        build_press_time_figure(state),
        build_pie_figure(overall_stats),
    ]


def save_songs_path(path):
    cfg = load_config()
    cfg["songs_path"] = path
    save_config(cfg)
    exists = os.path.isdir(path)
    return gr.Markdown(f"已保存: {path}\n{'✅ 目录存在' if exists else '⛔ 目录不存在'}")


def build_ui():
    config = load_config()
    songs_path = config.get("songs_path", "")

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

        with gr.Row():
            dist_plot = gr.Plot(label="偏移分布直方图")
            press_time_plot = gr.Plot(label="按压时长分布 (PressTime)")

        pie_plot = gr.Plot(label="判定分布饼图")

        # Events
        analyze_btn.click(
            fn=run_analysis,
            inputs=[osr_input, osu_input, songs_path_input],
            outputs=[status, summary_table, detail_table, overall_table,
                     merged_table, comparison_table,
                     state, mode_radio, col_checkbox,
                     dist_plot, press_time_plot, pie_plot],
        )

        for trigger in [mode_radio, col_checkbox]:
            trigger.change(
                fn=build_histogram_figure,
                inputs=[state, mode_radio, col_checkbox],
                outputs=dist_plot,
            )

        save_path_btn.click(fn=save_songs_path, inputs=songs_path_input,
                            outputs=save_path_msg)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860)
