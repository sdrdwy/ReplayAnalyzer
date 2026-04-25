import struct
import json
import lzma


class GameMode:
    STANDARD = 0
    TAIKO = 1
    CTB = 2
    MANIA = 3

    NAMES = {
        STANDARD: "Standard",
        TAIKO: "Taiko",
        CTB: "Catch",
        MANIA: "Mania",
    }


class ModsBit:
    NO_FAIL = 1 << 0
    EASY = 1 << 1
    TOUCH_DEVICE = 1 << 2
    HIDDEN = 1 << 3
    HARD_ROCK = 1 << 4
    SUDDEN_DEATH = 1 << 5
    DOUBLE_TIME = 1 << 6
    RELAX = 1 << 7
    HALF_TIME = 1 << 8
    NIGHTCORE = 1 << 9
    FLASHLIGHT = 1 << 10
    AUTOPLAY = 1 << 11
    SPUN_OUT = 1 << 12
    AUTOPILOT = 1 << 13
    PERFECT = 1 << 14
    KEY4 = 1 << 15
    KEY5 = 1 << 16
    KEY6 = 1 << 17
    KEY7 = 1 << 18
    KEY8 = 1 << 19
    FADE_IN = 1 << 20
    RANDOM = 1 << 21
    CINEMA = 1 << 22
    TARGET_PRACTICE = 1 << 23
    KEY9 = 1 << 24
    COOP = 1 << 25
    KEY1 = 1 << 26
    KEY3 = 1 << 27
    KEY2 = 1 << 28
    SCORE_V2 = 1 << 29
    MIRROR = 1 << 30

    _ALL = [
        (NO_FAIL, "NoFail"),
        (EASY, "Easy"),
        (TOUCH_DEVICE, "TouchDevice"),
        (HIDDEN, "Hidden"),
        (HARD_ROCK, "HardRock"),
        (SUDDEN_DEATH, "SuddenDeath"),
        (DOUBLE_TIME, "DoubleTime"),
        (RELAX, "Relax"),
        (HALF_TIME, "HalfTime"),
        (NIGHTCORE, "Nightcore"),
        (FLASHLIGHT, "Flashlight"),
        (AUTOPLAY, "Autoplay"),
        (SPUN_OUT, "SpunOut"),
        (AUTOPILOT, "Autopilot"),
        (PERFECT, "Perfect"),
        (KEY4, "Key4"),
        (KEY5, "Key5"),
        (KEY6, "Key6"),
        (KEY7, "Key7"),
        (KEY8, "Key8"),
        (FADE_IN, "FadeIn"),
        (RANDOM, "Random"),
        (CINEMA, "Cinema"),
        (TARGET_PRACTICE, "TargetPractice"),
        (KEY9, "Key9"),
        (COOP, "Coop"),
        (KEY1, "Key1"),
        (KEY3, "Key3"),
        (KEY2, "Key2"),
        (SCORE_V2, "ScoreV2"),
        (MIRROR, "Mirror"),
    ]

    @staticmethod
    def names(mask):
        return [name for bit, name in ModsBit._ALL if mask & bit]


def _read_uleb128(data, offset):
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, offset
        shift += 7


def _read_string(data, offset):
    flag = data[offset]
    offset += 1
    if flag == 0x00:
        return "", offset
    if flag == 0x0b:
        length, offset = _read_uleb128(data, offset)
        string = data[offset : offset + length].decode("utf-8")
        offset += length
        return string, offset
    raise ValueError(f"Invalid string flag: 0x{flag:02x}")


class OsuReplay:
    def __init__(self):
        self.game_mode = 0
        self.game_version = 0
        self.beatmap_md5 = ""
        self.player_name = ""
        self.replay_md5 = ""
        self.count_300 = 0
        self.count_100 = 0
        self.count_50 = 0
        self.count_geki = 0
        self.count_katu = 0
        self.count_miss = 0
        self.total_score = 0
        self.max_combo = 0
        self.perfect_combo = False
        self.mods = 0
        self.life_bar_graph = ""
        self.timestamp = 0
        self.compressed_data_length = 0
        self._compressed_data = b""
        self.replay_frames = []
        self.seed = None
        self.online_score_id = 0
        self.additional_mod_info = None

    @classmethod
    def from_bytes(cls, data: bytes):
        replay = cls()
        offset = 0

        replay.game_mode = data[offset]
        offset += 1

        replay.game_version = struct.unpack_from("<i", data, offset)[0]
        offset += 4

        replay.beatmap_md5, offset = _read_string(data, offset)
        replay.player_name, offset = _read_string(data, offset)
        replay.replay_md5, offset = _read_string(data, offset)

        replay.count_300 = struct.unpack_from("<h", data, offset)[0]
        offset += 2
        replay.count_100 = struct.unpack_from("<h", data, offset)[0]
        offset += 2
        replay.count_50 = struct.unpack_from("<h", data, offset)[0]
        offset += 2
        replay.count_geki = struct.unpack_from("<h", data, offset)[0]
        offset += 2
        replay.count_katu = struct.unpack_from("<h", data, offset)[0]
        offset += 2
        replay.count_miss = struct.unpack_from("<h", data, offset)[0]
        offset += 2

        replay.total_score = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        replay.max_combo = struct.unpack_from("<h", data, offset)[0]
        offset += 2
        replay.perfect_combo = bool(data[offset])
        offset += 1

        replay.mods = struct.unpack_from("<i", data, offset)[0]
        offset += 4

        replay.life_bar_graph, offset = _read_string(data, offset)

        replay.timestamp = struct.unpack_from("<q", data, offset)[0]
        offset += 8

        replay.compressed_data_length = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        replay._compressed_data = data[offset : offset + replay.compressed_data_length]
        offset += replay.compressed_data_length

        replay.online_score_id = struct.unpack_from("<q", data, offset)[0]
        offset += 8

        if replay.mods & ModsBit.TARGET_PRACTICE:
            replay.additional_mod_info = struct.unpack_from("<d", data, offset)[0]

        replay._parse_replay_data()
        return replay

    @classmethod
    def from_file(cls, path):
        with open(path, "rb") as f:
            return cls.from_bytes(f.read())

    def _parse_replay_data(self):
        if not self._compressed_data:
            return
        raw = lzma.decompress(self._compressed_data)
        text = raw.decode("utf-8")
        text = text.replace("|", ",")
        parts = text.split(",")
        i = 0
        while i + 3 < len(parts):
            w = int(parts[i])
            x = float(parts[i + 1])
            y = float(parts[i + 2])
            z = int(parts[i + 3])
            i += 4
            if w == -12345:
                self.seed = z
                continue
            self.replay_frames.append({
                "time_delta": w,
                "x": x,
                "y": y,
                "keys": z,
            })

    def to_dict(self):
        return {
            "game_mode": {
                "value": self.game_mode,
                "name": GameMode.NAMES.get(self.game_mode, f"Unknown ({self.game_mode})"),
            },
            "game_version": self.game_version,
            "beatmap_md5": self.beatmap_md5,
            "player_name": self.player_name,
            "replay_md5": self.replay_md5,
            "count_300": self.count_300,
            "count_100": self.count_100,
            "count_50": self.count_50,
            "count_geki": self.count_geki,
            "count_katu": self.count_katu,
            "count_miss": self.count_miss,
            "total_score": self.total_score,
            "max_combo": self.max_combo,
            "perfect_combo": self.perfect_combo,
            "mods": {
                "value": self.mods,
                "names": ModsBit.names(self.mods),
            },
            "life_bar_graph": self.life_bar_graph,
            "timestamp": self.timestamp,
            "compressed_data_length": self.compressed_data_length,
            "replay_frame_count": len(self.replay_frames),
            "seed": self.seed,
            "first_10_replay_frames": self.replay_frames[:10],
            "online_score_id": self.online_score_id,
            "additional_mod_info": self.additional_mod_info,
        }

    def to_json(self, indent=2, **kwargs):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, **kwargs)
