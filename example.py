import sys
import json
from collections import Counter
from osr_parser import OsuReplay


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_osr>", file=sys.stderr)
        sys.exit(1)

    replay = OsuReplay.from_file(sys.argv[1])
    result = replay.to_dict()

    frames = replay.replay_frames
    w_values = [f["time_delta"] for f in frames]

    result["unique_stats"] = {
        "x_count": len({f["x"] for f in frames}),
        "x_counts": dict(sorted(Counter(f["x"] for f in frames).items())),
        "y_count": len({f["y"] for f in frames}),
        "y_counts": dict(sorted(Counter(f["y"] for f in frames).items())),
        "keys_count": len({f["keys"] for f in frames}),
        "keys_counts": dict(sorted(Counter(f["keys"] for f in frames).items())),
        "w_total_count": len(w_values),
        "w_average": sum(w_values) / len(w_values) if w_values else 0,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
