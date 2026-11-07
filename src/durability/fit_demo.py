"""SYNTHETIC FIT files for demonstrating the local tool. They hold no real person's data and no research results."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FIT_EPOCH_S = 631065600  # 1989-12-31 UTC, set by the FIT format
METRES_PER_DEGREE = 111_000.0
ENUM, UINT8, UINT16, SINT32, UINT32 = 0x00, 0x02, 0x84, 0x85, 0x86
SIZES = {ENUM: 1, UINT8: 1, UINT16: 2, SINT32: 4, UINT32: 4}
FILE_ID, SESSION, RECORD = 0, 18, 20
SPORT_RUNNING, SUB_SPORT_GENERIC, FILE_TYPE_ACTIVITY = 1, 0, 4
CRC_TABLE = (0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401, 0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400)


def _crc(data: bytes) -> int:
    crc = 0
    for byte in data:
        for nibble in (byte & 0xF, byte >> 4):
            low = CRC_TABLE[crc & 0xF]
            crc = (crc >> 4) & 0x0FFF
            crc = crc ^ low ^ CRC_TABLE[nibble]
    return crc


def _definition(local: int, global_id: int, fields: list[tuple[int, int]]) -> bytes:
    head = struct.pack("<BBBHB", 0x40 | local, 0, 0, global_id, len(fields))
    return head + b"".join(struct.pack("<BBB", number, SIZES[kind], kind) for number, kind in fields)


def write_fit(path: str | Path, records: pd.DataFrame) -> Path:
    """Write one running activity as a valid FIT file with the record fields the tool reads."""
    stamps = records["timestamp"].map(lambda t: int(t.timestamp())).to_numpy()
    fit_times = stamps - FIT_EPOCH_S
    to_semicircles = 2**31 / 180.0
    body = bytearray(_definition(0, FILE_ID, [(0, ENUM)]))
    body += struct.pack("<BB", 0, FILE_TYPE_ACTIVITY)
    body += _definition(1, RECORD, [(253, UINT32), (0, SINT32), (1, SINT32), (2, UINT16), (3, UINT8)])
    for when, lat, lon, altitude, heart in zip(fit_times, records["latitude"], records["longitude"], records["altitude"], records["heart_rate"]):
        body += struct.pack("<BIiiHB", 1, int(when), round(lat * to_semicircles), round(lon * to_semicircles), round((altitude + 500) * 5), int(np.clip(round(heart), 1, 250)))
    body += _definition(2, SESSION, [(253, UINT32), (2, UINT32), (5, ENUM), (6, ENUM)])
    body += struct.pack("<BIIBB", 2, int(fit_times[-1]), int(fit_times[0]), SPORT_RUNNING, SUB_SPORT_GENERIC)
    header = struct.pack("<BBHI4s", 14, 0x20, 2132, len(body), b".FIT")
    header += struct.pack("<H", _crc(header))
    content = header + bytes(body)
    path = Path(path)
    path.write_bytes(content + struct.pack("<H", _crc(content)))
    return path


def synthetic_records(scenario: dict[str, Any], start: pd.Timestamp, seed: int, settings: dict[str, Any]) -> pd.DataFrame:
    """Flat 1 Hz run on a straight line, with heart rate from lagged speed plus a set drift."""
    rng = np.random.default_rng(seed)
    n = int(scenario["minutes"]) * 60
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = float(settings["speed_noise_ar1"]) * noise[i - 1] + rng.normal()
    speed = np.clip(float(settings["base_speed_ms"]) * (1 + float(scenario["cv"]) * noise / noise.std()), float(settings["min_speed_ms"]), float(settings["max_speed_ms"]))
    lagged = pd.Series(speed).ewm(alpha=1 / float(settings["hr_lag_s"])).mean().to_numpy()
    heart = float(settings["hr_rest_component"]) + float(settings["hr_speed_slope"]) * lagged + float(scenario["drift"]) * np.arange(n) / 3600 + rng.normal(0, float(settings["hr_noise_sd"]), n)
    frame = pd.DataFrame({
        "timestamp": pd.date_range(start, periods=n, freq="s"),
        "heart_rate": heart,
        "altitude": float(settings["altitude_m"]),
        "latitude": float(settings["start_lat_deg"]) + np.cumsum(speed) / METRES_PER_DEGREE,
        "longitude": float(settings["start_lon_deg"]),
    })
    for minute in scenario.get("pauses_at_min", []):
        begin = int(minute) * 60
        frame = frame[~frame.index.isin(range(begin, begin + int(scenario["pause_min"]) * 60))]
    return frame.reset_index(drop=True)


def build_demo_folder(folder: str | Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Write every configured scenario and return them with their file names, in date order."""
    settings = config["fit_demo"]
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(settings["start_date"], tz="UTC") + pd.Timedelta(hours=6)
    written = []
    for i, scenario in enumerate(settings["scenarios"], start=1):
        day = start + pd.Timedelta(days=(i - 1) * int(settings["days_between_runs"]))
        name = f"{i:02d}_{scenario['name']}.fit"
        write_fit(folder / name, synthetic_records(scenario, day, int(settings["seed"]) + i, settings))
        written.append({**scenario, "file": name})
    return written
