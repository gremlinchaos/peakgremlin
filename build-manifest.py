#!/usr/bin/env python3
"""Build WikiLocs track + curated moving-image manifest for the 3D page."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent
GPX = ROOT / "001_GPX" / "peakgremlingpx_wikilocs.gpx"
COROS = ROOT / "001_GPX" / "peakgremlingpx_coros.gpx"
LOCAL = timezone(timedelta(hours=2))

# Moving-image only, spaced along the climb (local HHMMSS)
PICKS = [
    "141235",  # lifts meeting
    "144947",
    "150900",
    "152504",
    "154219",
    "155356",
    "160514",  # summit
]


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_coros_hr():
    """Heart rate lives on the COROS export — match later by timestamp."""
    tree = ET.parse(COROS)
    series = []
    for trkpt in tree.findall(".//{*}trkpt"):
        t_el = trkpt.find("{*}time")
        if t_el is None:
            continue
        hr = None
        for child in trkpt.iter():
            if child.tag.split("}")[-1].lower() == "hr" and child.text:
                hr = int(float(child.text))
                break
        if hr is None:
            continue
        t = datetime.fromisoformat(t_el.text.replace("Z", "+00:00")).timestamp()
        series.append((t, hr))
    return series


def nearest_hr(series, ts: float) -> int | None:
    if not series:
        return None
    lo, hi = 0, len(series) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if series[mid][0] < ts:
            lo = mid + 1
        else:
            hi = mid
    i = lo
    if i > 0 and abs(series[i - 1][0] - ts) <= abs(series[i][0] - ts):
        i -= 1
    return series[i][1]


def load_track():
    tree = ET.parse(GPX)
    pts = []
    for trkpt in tree.findall(".//{*}trkpt"):
        t_el = trkpt.find("{*}time")
        e_el = trkpt.find("{*}ele")
        if t_el is None:
            continue
        t = datetime.fromisoformat(t_el.text.replace("Z", "+00:00")).timestamp()
        ele = float(e_el.text) if e_el is not None and e_el.text else 0.0
        pts.append(
            {
                "t": t,
                "lat": float(trkpt.get("lat")),
                "lon": float(trkpt.get("lon")),
                "ele": ele,
            }
        )

    hr_series = load_coros_hr()
    for p in pts:
        p["hr"] = nearest_hr(hr_series, p["t"])

    # speed + cumulative distance
    dist = 0.0
    pts[0]["speed"] = 0.0
    pts[0]["dist"] = 0.0
    for i in range(1, len(pts)):
        d = haversine_m(pts[i - 1]["lat"], pts[i - 1]["lon"], pts[i]["lat"], pts[i]["lon"])
        dt = max(pts[i]["t"] - pts[i - 1]["t"], 1e-3)
        dist += d
        pts[i]["dist"] = dist
        pts[i]["speed"] = d / dt  # m/s

    # light smooth on speed
    for i in range(1, len(pts) - 1):
        pts[i]["speed"] = (
            pts[i - 1]["speed"] + pts[i]["speed"] + pts[i + 1]["speed"]
        ) / 3

    return pts


def nearest(pts, ts: float):
    lo, hi = 0, len(pts) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if pts[mid]["t"] < ts:
            lo = mid + 1
        else:
            hi = mid
    i = lo
    if i > 0 and abs(pts[i - 1]["t"] - ts) <= abs(pts[i]["t"] - ts):
        i -= 1
    return i, pts[i]


def scan_videos(pts):
    start, end = pts[0]["t"], pts[-1]["t"]
    pat = re.compile(r"peakgremlin_movingimage_(\d{8})_(\d{6})")
    out = []
    for path in sorted((ROOT / "002_MOVING_IMAGE").iterdir()):
        if path.name.startswith("."):
            continue
        m = pat.search(path.name)
        if not m:
            continue
        local = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(
            tzinfo=LOCAL
        )
        ts = local.astimezone(timezone.utc).timestamp()
        if not (start <= ts <= end):
            continue
        i, p = nearest(pts, ts)
        out.append(
            {
                "file": path.name,
                "src": f"002_MOVING_IMAGE/{path.name}",
                "type": "video",
                "t": ts,
                "local": local,
                "ele": p["ele"],
                "speed": p["speed"],
                "hr": p.get("hr"),
                "dist": p["dist"],
                "lat": p["lat"],
                "lon": p["lon"],
                "index": i,
            }
        )
    return out


def pick_video(videos, hhmmss: str):
    exact = [v for v in videos if v["local"].strftime("%H%M%S") == hhmmss]
    if exact:
        return exact[0]
    target = datetime.strptime("20261002" + hhmmss, "%Y%m%d%H%M%S").replace(
        tzinfo=LOCAL
    ).timestamp()
    return min(videos, key=lambda v: abs(v["t"] - target))


def downsample(pts, n=480):
    if len(pts) <= n:
        sample = pts
    else:
        step = (len(pts) - 1) / (n - 1)
        sample = [pts[int(round(i * step))] for i in range(n)]

    ele_min = min(p["ele"] for p in pts)
    ele_max = max(p["ele"] for p in pts)
    spd_max = max(p["speed"] for p in pts) or 1.0
    lat0 = sum(p["lat"] for p in sample) / len(sample)
    lon0 = sum(p["lon"] for p in sample) / len(sample)
    # local meters projection
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))

    out = []
    for p in sample:
        x = (p["lon"] - lon0) * m_per_deg_lon
        z = -((p["lat"] - lat0) * m_per_deg_lat)
        y = p["ele"] - ele_min
        out.append(
            {
                "t": datetime.fromtimestamp(p["t"], timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "lat": round(p["lat"], 6),
                "lon": round(p["lon"], 6),
                "ele": round(p["ele"], 1),
                "speed": round(p["speed"], 3),
                "dist": round(p["dist"], 1),
                "eleN": round((p["ele"] - ele_min) / (ele_max - ele_min), 4),
                "spdN": round(min(p["speed"] / spd_max, 1.0), 4),
                "hr": p.get("hr"),
                "x": round(x, 2),
                "y": round(y, 2),
                "z": round(z, 2),
            }
        )
    return out, ele_min, ele_max, spd_max, lat0, lon0


def main() -> None:
    pts = load_track()
    videos = scan_videos(pts)
    path, ele_min, ele_max, spd_max, lat0, lon0 = downsample(pts)

    curated = []
    used = set()
    for hhmmss in PICKS:
        v = pick_video(videos, hhmmss)
        if v["file"] in used:
            continue
        used.add(v["file"])
        curated.append(v)
    curated.sort(key=lambda a: a["t"])

    assets = []
    for a in curated:
        # progress along full track 0..1 by time
        prog = (a["t"] - pts[0]["t"]) / (pts[-1]["t"] - pts[0]["t"])
        assets.append(
            {
                "file": a["file"],
                "src": a["src"],
                "type": "video",
                "time": datetime.fromtimestamp(a["t"], timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "displayTime": a["local"].strftime("%H:%M"),
                "ele": round(a["ele"]),
                "eleN": round((a["ele"] - ele_min) / (ele_max - ele_min), 4),
                "speed": round(a["speed"], 2),
                "spdN": round(min(a["speed"] / spd_max, 1.0), 4),
                "hr": a.get("hr"),
                "lat": a["lat"],
                "lon": a["lon"],
                "progress": round(prog, 4),
            }
        )

    out = {
        "gpx": "001_GPX/peakgremlingpx_wikilocs.gpx",
        "title": "Peak Gremlin",
        "track": {
            "name": "Passo Falzarego — Rifugio Lagazuoi",
            "start": datetime.fromtimestamp(pts[0]["t"], timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "end": datetime.fromtimestamp(pts[-1]["t"], timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "eleMin": ele_min,
            "eleMax": ele_max,
            "spdMax": round(spd_max, 3),
            "distance": round(pts[-1]["dist"], 1),
            "center": {"lat": lat0, "lon": lon0},
        },
        "path": path,
        "assets": assets,
    }
    path_out = ROOT / "manifest.json"
    path_out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {path_out.name}: {len(path)} path pts, {len(assets)} videos")
    for a in assets:
        print(
            f"  {a['displayTime']}  {a['ele']}m  {a.get('hr')}bpm  {a['speed']}m/s  p={a['progress']}  {a['file']}"
        )


if __name__ == "__main__":
    main()
