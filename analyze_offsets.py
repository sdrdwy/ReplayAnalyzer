import sys
import os
import json
import math
import lzma
from statistics import mean, median, pvariance, pstdev
from osr_parser import OsuReplay, GameMode, ModsBit
from osu import Osu


JUDGMENT_NAMES = ["PERFECT", "GREAT", "GOOD", "OK", "MEH", "MISS"]


def compute_windows(od):
    perfect = 16
    great = int(64 - 3 * od)
    good = int(97 - 3 * od)
    ok = int(127 - 3 * od)
    meh = int(151 - 3 * od)
    miss = int(188 - 3 * od)
    return {"perfect": perfect, "great": great, "good": good,
            "ok": ok, "meh": meh, "miss": miss}


def classify_judgment(offset_ms, windows):
    abs_offset = abs(offset_ms)
    if abs_offset <= windows["perfect"]:
        return "PERFECT"
    if abs_offset <= windows["great"]:
        return "GREAT"
    if abs_offset <= windows["good"]:
        return "GOOD"
    if abs_offset <= windows["ok"]:
        return "OK"
    if abs_offset <= windows["meh"]:
        return "MEH"
    if abs_offset <= windows["miss"]:
        return "MISS"
    return None


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
    for h in o.data["[HitObjects]"][:]:
        x = int(h[0])
        time = int(h[2])
        type_ = int(h[3])
        col = math.floor(x * cs / 512)
        if col < 0 or col >= cs:
            continue
        is_hold = bool(type_ & 128)
        if is_hold:
            end_time = int(h[5])
            notes.append({
                "column": col,
                "time": time,
                "type": "hold",
                "end_time": end_time,
            })
        else:
            notes.append({
                "column": col,
                "time": time,
                "type": "tap",
                "end_time": None,
            })

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

    raw_frames = text.split(",")
    frames = []
    cum_time = 0
    for rf in raw_frames:
        if not rf.strip():
            continue
        parts = rf.split("|")
        if len(parts) != 4:
            continue
        w = int(parts[0])
        if w == -12345:
            continue
        x_bitmask = int(parts[1])
        cum_time += w
        if x_bitmask == 256 or cum_time < 0:
            continue
        frames.append({
            "time": cum_time,
            "key_state": x_bitmask,
        })

    return player, mods_value, mods_names, frames


def detect_events(frames, cs):
    if not frames:
        return [], []

    prev_state = frames[0]["key_state"]
    press_events = []
    release_events = []
    for f in frames[1:]:
        curr_state = f["key_state"]
        changed = prev_state ^ curr_state
        if changed:
            for col in range(cs):
                if changed & (1 << col):
                    is_press = (curr_state >> col) & 1
                    ev = {"time": f["time"], "column": col}
                    if is_press:
                        press_events.append(ev)
                    else:
                        release_events.append(ev)
        prev_state = curr_state
    return press_events, release_events


def _two_pointer_match(events, targets, max_dist):
    events = sorted(events)
    targets = sorted(targets)
    matches = []
    ei = 0
    used_events = set()
    ti = 0
    while ti < len(targets) and ei < len(events):
        t = targets[ti]
        while ei + 1 < len(events) and abs(events[ei + 1] - t) < abs(events[ei] - t):
            ei += 1
        if abs(events[ei] - t) <= max_dist:
            matches.append((t, events[ei]))
            used_events.add(ei)
            ei += 1
        else:
            pass
        ti += 1
    unused = [e for i, e in enumerate(events) if i not in used_events]
    return matches, unused


def match_events(notes, press_events, release_events, cs, max_match_dist):
    press_by_col = {c: sorted([p["time"] for p in press_events if p["column"] == c])
                    for c in range(cs)}
    release_by_col = {c: sorted([r["time"] for r in release_events if r["column"] == c])
                      for c in range(cs)}

    notes_by_col = {c: [] for c in range(cs)}
    for n in notes:
        notes_by_col[n["column"]].append(n)

    offsets = []
    total_unused_press = 0
    total_unused_release = 0

    for col in range(cs):
        col_notes = notes_by_col[col]
        tap_head_notes = [n["time"] for n in col_notes]
        head_matches, unused_p = _two_pointer_match(
            press_by_col.get(col, []), tap_head_notes, max_match_dist)
        total_unused_press += len(unused_p)

        hold_notes = [n for n in col_notes if n["type"] == "hold"]
        tail_targets = [n["end_time"] for n in hold_notes]
        tail_matches, unused_r = _two_pointer_match(
            release_by_col.get(col, []), tail_targets, max_match_dist)
        total_unused_release += len(unused_r)

        head_idx = 0
        tail_idx = 0
        for n in col_notes:
            expected = n["time"]
            is_hold = n["type"] == "hold"

            if head_idx < len(head_matches) and head_matches[head_idx][0] == expected:
                _, press_time = head_matches[head_idx]
                offsets.append({
                    "column": col, "type": "tap" if not is_hold else "hold_head",
                    "offset_ms": press_time - expected,
                    "hit_time_ms": expected, "event_time_ms": press_time,
                })
                head_idx += 1
            else:
                offsets.append({
                    "column": col, "type": "tap" if not is_hold else "hold_head",
                    "offset_ms": None,
                    "hit_time_ms": expected, "event_time_ms": None,
                })

            if is_hold:
                tail_expected = n["end_time"]
                if tail_idx < len(tail_matches) and tail_matches[tail_idx][0] == tail_expected:
                    _, release_time = tail_matches[tail_idx]
                    offsets.append({
                        "column": col, "type": "hold_tail",
                        "offset_ms": release_time - tail_expected,
                        "hit_time_ms": tail_expected, "event_time_ms": release_time,
                    })
                    tail_idx += 1
                else:
                    offsets.append({
                        "column": col, "type": "hold_tail",
                        "offset_ms": None,
                        "hit_time_ms": tail_expected, "event_time_ms": None,
                    })

    total_press_events = sum(len(v) for v in press_by_col.values())
    total_release_events = sum(len(v) for v in release_by_col.values())

    extras = {
        "extra_presses": total_unused_press,
        "extra_releases": total_unused_release,
    }

    return offsets, extras


def build_statistics(offsets, windows):
    grouped = {}
    for o in offsets:
        if o["offset_ms"] is None:
            continue
        key = (o["type"], classify_judgment(o["offset_ms"], windows))
        grouped.setdefault(key, []).append(o["offset_ms"])

    by_type = {}
    for (typ, jdg), vals in grouped.items():
        by_type.setdefault(typ, {})[jdg] = {
            "count": len(vals),
            "mean": round(mean(vals), 4),
            "median": round(median(vals), 4),
            "variance": round(pvariance(vals), 4),
            "std": round(pstdev(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    by_column = []
    for col in sorted(set(o["column"] for o in offsets if o["offset_ms"] is not None)):
        col_offsets = [o for o in offsets if o["column"] == col and o["offset_ms"] is not None]
        col_data = {"column": col}
        for typ in ["tap", "hold_head", "hold_tail"]:
            typ_offsets = [o["offset_ms"] for o in col_offsets if o["type"] == typ]
            if not typ_offsets:
                continue
            jdg_counts = {}
            for v in typ_offsets:
                j = classify_judgment(v, windows)
                jdg_counts[j] = jdg_counts.get(j, 0) + 1
            col_data[typ] = {
                "count": len(typ_offsets),
                "mean": round(mean(typ_offsets), 4),
                "median": round(median(typ_offsets), 4),
                "variance": round(pvariance(typ_offsets), 4),
                "std": round(pstdev(typ_offsets), 4),
                "min": round(min(typ_offsets), 4),
                "max": round(max(typ_offsets), 4),
                "judgment_counts": jdg_counts,
            }
        by_column.append(col_data)

    overall = {}
    for typ in ["tap", "hold_head", "hold_tail"]:
        vals = [o["offset_ms"] for o in offsets if o["type"] == typ and o["offset_ms"] is not None]
        if not vals:
            continue
        jdg_counts = {}
        for v in vals:
            j = classify_judgment(v, windows)
            jdg_counts[j] = jdg_counts.get(j, 0) + 1
        overall[typ] = {
            "count": len(vals),
            "mean": round(mean(vals), 4),
            "median": round(median(vals), 4),
            "variance": round(pvariance(vals), 4),
            "std": round(pstdev(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "judgment_counts": jdg_counts,
        }

    return overall, by_type, by_column


def build_output(meta, player, mods_value, mods_names, cs, od, windows,
                 notes, offsets, match_extras,
                 overall_stats, by_type_stats, by_column_stats):
    total_notes = len(notes)
    matched_entries = sum(1 for o in offsets if o["offset_ms"] is not None)
    total_entries = len(offsets)

    offset_list = []
    for o in offsets:
        if o["offset_ms"] is not None:
            offset_list.append({
                "column": o["column"],
                "type": o["type"],
                "offset_ms": o["offset_ms"],
                "hit_time_ms": o["hit_time_ms"],
                "event_time_ms": o["event_time_ms"],
                "judgment": classify_judgment(o["offset_ms"], windows),
            })

    max_note_time = max(n["time"] for n in notes) if notes else 0

    return {
        "metadata": {
            "player": player,
            "mods": {
                "value": mods_value,
                "names": mods_names,
            },
            "beatmap": {
                "title": meta["title"],
                "title_unicode": meta["title_unicode"],
                "artist": meta["artist"],
                "artist_unicode": meta["artist_unicode"],
                "creator": meta["creator"],
                "version": meta["version"],
                "beatmap_id": meta["beatmap_id"],
                "beatmap_set_id": meta["beatmap_set_id"],
            },
            "cs": cs,
            "od": od,
            "mode": "mania",
        },
        "hit_windows": windows,
        "time_range_ms": {
            "first_note": notes[0]["time"] if notes else 0,
            "last_note": max_note_time,
        },
        "match_summary": {
            "total_notes": total_notes,
            "total_entries": total_entries,
            "matched_entries": matched_entries,
            "extra_presses": match_extras["extra_presses"],
            "extra_releases": match_extras["extra_releases"],
        },
        "statistics": {
            "overall_by_type": overall_stats,
            "by_column": by_column_stats,
        },
        "judgment_distribution": by_type_stats,
        "offsets": offset_list,
    }


def build_filename(meta, player):
    safe_player = player.replace(" ", "_")
    safe_artist = meta["artist"].replace(" ", "_")
    safe_title = meta["title"].replace(" ", "_")
    safe_version = meta["version"].replace(" ", "_")
    return f"{safe_player}_{safe_artist}_-_{safe_title}_{safe_version}_offsets.json"


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <replay.osr> <beatmap.osu> [output.json]",
              file=sys.stderr)
        sys.exit(1)

    osr_path = sys.argv[1]
    osu_path = sys.argv[2]

    if not os.path.exists(osr_path):
        print(f"Error: replay file not found: {osr_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(osu_path):
        print(f"Error: beatmap file not found: {osu_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing replay: {osr_path}")
    player, mods_value, mods_names, frames = parse_replay_frames(osr_path)
    cs_from_osu = None

    print(f"Parsing beatmap: {osu_path}")
    notes, cs, od, meta = parse_beatmap(osu_path)
    cs_from_osu = cs

    windows = compute_windows(od)

    print(f"Player: {player}  |  CS={cs}  |  OD={od}")
    print(f"Notes: {len(notes)}  |  Frames: {len(frames)}")
    print(f"Mods: {mods_names}")

    press_events, release_events = detect_events(frames, cs)
    print(f"Press events: {len(press_events)}  |  Release events: {len(release_events)}")

    max_match_dist = windows["miss"]
    print(f"Max matching distance: {max_match_dist}ms (miss window)")

    offsets, match_extras = match_events(notes, press_events, release_events, cs, max_match_dist)

    overall_stats, by_type_stats, by_column_stats = build_statistics(offsets, windows)

    result = build_output(
        meta, player, mods_value, mods_names, cs, od, windows,
        notes, offsets, match_extras,
        overall_stats, by_type_stats, by_column_stats,
    )

    if len(sys.argv) >= 4:
        out_path = sys.argv[3]
    else:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)
        out_path = os.path.join(data_dir, build_filename(meta, player))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    me = sum(1 for o in offsets if o["offset_ms"] is not None)
    print(f"\nOutput: {out_path}")
    print(f"Matched entries: {me}/{len(offsets)}  |  "
          f"Extra presses: {match_extras['extra_presses']}  |  "
          f"Extra releases: {match_extras['extra_releases']}")

    for typ in ["tap", "hold_head", "hold_tail"]:
        if typ in overall_stats:
            s = overall_stats[typ]
            print(f"  {typ:>10}: count={s['count']:>4}  "
                  f"mean={s['mean']:>8.2f}  "
                  f"median={s['median']:>8.2f}  "
                  f"std={s['std']:>8.2f}")


if __name__ == "__main__":
    main()
