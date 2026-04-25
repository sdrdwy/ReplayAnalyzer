import sys
import os
import json
import math
import lzma
from statistics import mean, median, pvariance, pstdev
from osr_parser import OsuReplay, GameMode, ModsBit
from osu import Osu


JUDGMENT_ORDER = ["PERFECT", "GREAT", "GOOD", "OK", "MEH", "MISS"]


def map_difficulty_range(difficulty, min_val, mid_val, max_val):
    if difficulty > 5:
        return mid_val + (max_val - mid_val) * (difficulty - 5) / 5
    if difficulty < 5:
        return mid_val - (mid_val - min_val) * (5 - difficulty) / 5
    return mid_val


def compute_windows(od, scorev2=False):
    if scorev2:
        perfect = map_difficulty_range(od, 22.4, 19.4, 13.9)
        great = map_difficulty_range(od, 64.0, 49.0, 34.0)
        good = map_difficulty_range(od, 97.0, 82.0, 67.0)
        ok = map_difficulty_range(od, 127.0, 112.0, 97.0)
        meh = map_difficulty_range(od, 151.0, 136.0, 121.0)
        miss = map_difficulty_range(od, 188.0, 173.0, 158.0)
    else:
        perfect = 16
        great = int(64 - 3 * od)
        good = int(97 - 3 * od)
        ok = int(127 - 3 * od)
        meh = int(151 - 3 * od)
        miss = int(188 - 3 * od)
    return {"perfect": perfect, "great": great, "good": good,
            "ok": ok, "meh": meh, "miss": miss}


def classify_judgment(offset_ms, windows):
    ab = abs(offset_ms)
    if ab <= windows["perfect"]: return "PERFECT"
    if ab <= windows["great"]: return "GREAT"
    if ab <= windows["good"]: return "GOOD"
    if ab <= windows["ok"]: return "OK"
    if ab <= windows["meh"]: return "MEH"
    if ab <= windows["miss"]: return "MISS"
    return None


def classify_combined_hold(head_offset, tail_offset, windows):
    total = abs(head_offset) + abs(tail_offset)
    h = abs(head_offset)
    if h <= windows["perfect"] * 1.2 and total <= windows["perfect"] * 2.4:
        return "PERFECT"
    if h <= windows["great"] * 1.1 and total <= windows["great"] * 2.2:
        return "GREAT"
    if h <= windows["good"] and total <= windows["good"] * 2:
        return "GOOD"
    if h <= windows["ok"] and total <= windows["ok"] * 2:
        return "OK"
    if h <= windows["meh"] or total <= windows["meh"] * 2:
        return "MEH"
    return "MISS"


def parse_beatmap(osu_path):
    o = Osu()
    o.load_from_file(osu_path)
    cs = int(o.data["[Difficulty]"]["CircleSize"])
    od = float(o.data["[Difficulty]"]["OverallDifficulty"])
    meta = {
        "title": o.data["[Metadata]"].get("Title", ""),
        "title_unicode": o.data["[Metadata]"].get("TitleUnicode", ""),
        "artist": o.data["[Metadata]"].get("Artist", ""),
        "artist_unicode": o.data["[Metadata]"].get("ArtistUnicode", ""),
        "creator": o.data["[Metadata]"].get("Creator", ""),
        "version": o.data["[Metadata]"].get("Version", ""),
        "beatmap_id": o.data["[Metadata]"].get("BeatmapID", ""),
        "beatmap_set_id": o.data["[Metadata]"].get("BeatmapSetID", ""),
    }
    notes = []
    for h in o.data["[HitObjects]"]:
        x_pos = int(h[0]); time = int(h[2]); type_ = int(h[3])
        col = math.floor(x_pos * cs / 512)
        if col < 0 or col >= cs: continue
        is_hold = bool(type_ & 128)
        if is_hold:
            end_time = int(h[5])
            notes.append({"column": col, "time": time, "type": "hold", "end_time": end_time})
        else:
            notes.append({"column": col, "time": time, "type": "tap", "end_time": None})
    notes.sort(key=lambda n: n["time"])
    return notes, cs, od, meta


def parse_replay_frames(osr_path):
    replay = OsuReplay.from_file(osr_path)
    if replay.game_mode != GameMode.MANIA:
        raise NotImplementedError(
            f"Non-mania mode (game_mode={replay.game_mode}) not supported")
    player = replay.player_name
    mods_value = replay.mods
    mods_names = ModsBit.names(mods_value)
    raw = lzma.decompress(replay._compressed_data)
    text = raw.decode("utf-8")
    frames = []
    cum = 0
    for rf in text.split(","):
        if not rf.strip(): continue
        parts = rf.split("|")
        if len(parts) != 4: continue
        w = int(parts[0]); x = int(parts[1])
        if w == -12345: continue
        cum += w
        if x == 256 or cum < 0: continue
        frames.append({"time": cum, "key_state": x})
    return player, mods_value, mods_names, frames


def extract_actions(frames, cs):
    hold_begin = {c: None for c in range(cs)}
    actions = []
    if not frames: return actions
    prev_x = frames[0]["key_state"]
    for f in frames[1:]:
        x = f["key_state"]
        if x == prev_x: continue
        changed = prev_x ^ x
        for col in range(cs):
            if changed & (1 << col):
                is_press = (x >> col) & 1
                t = f["time"]
                if is_press:
                    hold_begin[col] = t
                else:
                    if hold_begin[col] is not None:
                        actions.append({
                            "time": hold_begin[col],
                            "column": col,
                            "end_time": t,
                            "duration": t - hold_begin[col],
                        })
                        hold_begin[col] = None
        prev_x = x
    return actions


def action_driven_judge(notes, actions, windows, cs, scorev2=False):
    """
    Reference: BaseJudger.judge() + OsuManiaJudger.canHit()
    Action-driven matching: for each action, scan notes in order for best match.
    """
    remaining = list(sorted(notes, key=lambda n: n["time"]))
    all_offsets = []
    missed_count = 0
    matched_action_count = 0

    for action in sorted(actions, key=lambda a: a["time"]):
        if not remaining:
            break

        # Scan remaining notes to find a match in the same column
        idx = 0
        while idx < len(remaining):
            note = remaining[idx]
            if note["column"] != action["column"]:
                idx += 1
                continue

            # TOO_EARLY: action before note's early-miss boundary
            if action["time"] < note["time"] - windows["miss"]:
                break

            # TOO_LATE: note can't be hit by this action, remove it
            if action["time"] > note["time"] + windows["miss"]:
                remaining.pop(idx)
                missed_count += 1
                all_offsets.append({
                    "column": note["column"], "type": "tap" if note["type"] == "tap" else "hold",
                    "offset_ms": None, "hit_time_ms": note["time"], "event_time_ms": None,
                })
                continue

            # HIT: action matches note
            remaining.pop(idx)
            matched_action_count += 1
            if note["type"] == "tap":
                offset = action["time"] - note["time"]
                all_offsets.append({
                    "column": note["column"], "type": "tap",
                    "offset_ms": offset, "hit_time_ms": note["time"], "event_time_ms": action["time"],
                })
            else:
                head_offset = action["time"] - note["time"]
                tail_offset = action["end_time"] - note["end_time"]
                if scorev2:
                    tail_win = {k: v * 1.5 for k, v in windows.items()}
                    all_offsets.append({
                        "column": note["column"], "type": "hold_head",
                        "offset_ms": head_offset, "hit_time_ms": note["time"], "event_time_ms": action["time"],
                    })
                    all_offsets.append({
                        "column": note["column"], "type": "hold_tail",
                        "offset_ms": tail_offset, "hit_time_ms": note["end_time"], "event_time_ms": action["end_time"],
                    })
                else:
                    all_offsets.append({
                        "column": note["column"], "type": "hold",
                        "offset_ms": head_offset, "hit_time_ms": note["time"], "event_time_ms": action["time"],
                        "head_offset": head_offset, "tail_offset": tail_offset,
                    })
            break  # action consumed, move to next action

    # Remaining notes = missed
    for note in remaining:
        missed_count += 1
        all_offsets.append({
            "column": note["column"], "type": "tap" if note["type"] == "tap" else "hold",
            "offset_ms": None, "hit_time_ms": note["time"], "event_time_ms": None,
        })

    return all_offsets, missed_count, len(actions) - matched_action_count


def build_statistics(offsets, windows):
    grouped = {}
    for o in offsets:
        if o["offset_ms"] is None: continue
        typ = o["type"]
        if typ == "hold":
            jdg = classify_combined_hold(o["head_offset"], o["tail_offset"], windows)
        else:
            jdg = classify_judgment(o["offset_ms"], windows)
        grouped.setdefault(typ, {}).setdefault(jdg, []).append(o["offset_ms"])

    def stat_row(vals):
        return {
            "count": len(vals),
            "mean": round(mean(vals), 4),
            "median": round(median(vals), 4),
            "variance": round(pvariance(vals), 4),
            "std": round(pstdev(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    overall = {}
    for typ in ["tap", "hold", "hold_head", "hold_tail"]:
        vals = [o["offset_ms"] for o in offsets if o["offset_ms"] is not None and o["type"] == typ]
        if not vals: continue
        jdg_counts = {}
        for o in offsets:
            if o["offset_ms"] is None or o["type"] != typ: continue
            if typ == "hold":
                j = classify_combined_hold(o["head_offset"], o["tail_offset"], windows)
            else:
                j = classify_judgment(o["offset_ms"], windows)
            jdg_counts[j] = jdg_counts.get(j, 0) + 1
        d = stat_row(vals)
        d["judgment_counts"] = jdg_counts
        overall[typ] = d

    by_column = []
    for col in sorted(set(o["column"] for o in offsets if o["offset_ms"] is not None)):
        col_data = {"column": col}
        for typ in ["tap", "hold", "hold_head", "hold_tail"]:
            typ_vals = [o for o in offsets if o["column"] == col and o["type"] == typ and o["offset_ms"] is not None]
            if not typ_vals: continue
            jdg_counts = {}
            for o in typ_vals:
                if typ == "hold":
                    j = classify_combined_hold(o["head_offset"], o["tail_offset"], windows)
                else:
                    j = classify_judgment(o["offset_ms"], windows)
                jdg_counts[j] = jdg_counts.get(j, 0) + 1
            vals = [o["offset_ms"] for o in typ_vals]
            d = stat_row(vals)
            d["judgment_counts"] = jdg_counts
            col_data[typ] = d
        by_column.append(col_data)

    by_type = {}
    for typ in ["tap", "hold", "hold_head", "hold_tail"]:
        typ_offsets = [o for o in offsets if o["type"] == typ and o["offset_ms"] is not None]
        if not typ_offsets: continue
        jdg_counts = {}
        for o in typ_offsets:
            if typ == "hold":
                j = classify_combined_hold(o["head_offset"], o["tail_offset"], windows)
            else:
                j = classify_judgment(o["offset_ms"], windows)
            jdg_counts[j] = jdg_counts.get(j, 0) + 1
        vals = [o["offset_ms"] for o in typ_offsets]
        d = stat_row(vals)
        d["judgment_counts"] = jdg_counts
        by_type[typ] = d

    return overall, by_type, by_column


def build_judgment_stats(notes, offsets, windows, scorev2=False):
    from collections import Counter
    result = Counter()
    if not scorev2:
        for o in offsets:
            if o["offset_ms"] is None: continue
            if o["type"] == "hold":
                result[classify_combined_hold(o["head_offset"], o["tail_offset"], windows)] += 1
            elif o["type"] in ("tap",):
                result[classify_judgment(o["offset_ms"], windows)] += 1
    else:
        tail_win = {k: v * 1.5 for k, v in windows.items()}
        for o in offsets:
            if o["offset_ms"] is None: continue
            if o["type"] == "hold_head":
                result[classify_judgment(o["offset_ms"], windows)] += 1
            elif o["type"] == "hold_tail":
                result[classify_judgment(o["offset_ms"], tail_win)] += 1
            elif o["type"] == "tap":
                result[classify_judgment(o["offset_ms"], windows)] += 1
    return dict(result)


def build_output(meta, player, mods_value, mods_names, cs, od, windows,
                 notes, offsets, match_extras, overall_stats, by_type_stats, by_column_stats):
    total_notes = len(notes)
    matched_entries = sum(1 for o in offsets if o["offset_ms"] is not None)
    total_entries = len(offsets)
    if isinstance(match_extras, dict):
        extra_p = match_extras.get("extra_presses", 0)
        extra_r = match_extras.get("extra_releases", 0)
    else:
        extra_p = match_extras[0] if len(match_extras) > 0 else 0
        extra_r = match_extras[1] if len(match_extras) > 1 else 0

    offset_list = []
    for o in offsets:
        if o["offset_ms"] is not None:
            offset_list.append({
                "column": o["column"],
                "type": o["type"],
                "offset_ms": o["offset_ms"],
                "hit_time_ms": o["hit_time_ms"],
                "event_time_ms": o["event_time_ms"],
            })
    max_note_time = max(n["time"] for n in notes) if notes else 0
    return {
        "metadata": {
            "player": player,
            "mods": {"value": mods_value, "names": mods_names},
            "beatmap": {
                "title": meta["title"], "title_unicode": meta["title_unicode"],
                "artist": meta["artist"], "artist_unicode": meta["artist_unicode"],
                "creator": meta["creator"], "version": meta["version"],
                "beatmap_id": meta["beatmap_id"], "beatmap_set_id": meta["beatmap_set_id"],
            },
            "cs": cs, "od": od, "mode": "mania",
        },
        "hit_windows": windows,
        "time_range_ms": {"first_note": notes[0]["time"] if notes else 0, "last_note": max_note_time},
        "match_summary": {
            "total_notes": total_notes, "total_entries": total_entries,
            "matched_entries": matched_entries,
            "extra_presses": extra_p, "extra_releases": extra_r,
        },
        "statistics": {"overall_by_type": overall_stats, "by_column": by_column_stats},
        "judgment_distribution": by_type_stats,
        "offsets": offset_list,
    }


def build_filename(meta, player):
    safe = lambda s: s.replace(" ", "_")
    return f"{safe(player)}_{safe(meta['artist'])}_-_{safe(meta['title'])}_{safe(meta['version'])}_offsets.json"


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <replay.osr> <beatmap.osu> [output.json]", file=sys.stderr)
        sys.exit(1)
    osr_path, osu_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(osr_path): print(f"Error: {osr_path} not found", file=sys.stderr); sys.exit(1)
    if not os.path.exists(osu_path): print(f"Error: {osu_path} not found", file=sys.stderr); sys.exit(1)

    print(f"Parsing replay: {osr_path}")
    player, mods_v, mods_n, frames = parse_replay_frames(osr_path)
    notes, cs, od, meta = parse_beatmap(osu_path)
    scorev2 = bool(mods_v & ModsBit.SCORE_V2)
    windows = compute_windows(od, scorev2=scorev2)

    print(f"{player}  CS={cs}  OD={od}  ScoreV2={scorev2}")
    print(f"Notes: {len(notes)}  Frames: {len(frames)}")

    actions = extract_actions(frames, cs)
    print(f"Actions: {len(actions)} (press+release)")

    offsets, missed, extra = action_driven_judge(notes, actions, windows, cs, scorev2=scorev2)
    overall_stats, by_type_stats, by_column_stats = build_statistics(offsets, windows)
    result = build_output(meta, player, mods_v, mods_n, cs, od, windows,
                          notes, offsets, (extra, 0), overall_stats, by_type_stats, by_column_stats)

    out_path = sys.argv[3] if len(sys.argv) >= 4 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", build_filename(meta, player))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    me = sum(1 for o in offsets if o["offset_ms"] is not None)
    print(f"Output: {out_path}")
    print(f"Matched: {me}/{len(offsets)}  Extra: {extra}")
    for typ in ["tap", "hold", "hold_head", "hold_tail"]:
        if typ in overall_stats:
            s = overall_stats[typ]
            print(f"  {typ:>10}: count={s['count']:>4}  mean={s['mean']:>8.2f}  median={s['median']:>8.2f}  std={s['std']:>8.2f}")


if __name__ == "__main__":
    main()
