#!/usr/bin/env python3
"""Convert GPS samples from a DJI ExifTool -ee text dump into a GPX 1.1 track.

Usage:
    python3 dji_exiftool_to_gpx.py DJI_video.txt
    python3 dji_exiftool_to_gpx.py DJI_video.txt -o flight.gpx

Only reads the text dump; never opens the original video or invokes ExifTool.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

GPX_NS = "http://www.topografix.com/GPX/1/1"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
SCHEMA_URL = "http://www.topografix.com/GPX/1/1/gpx.xsd"

ET.register_namespace("", GPX_NS)
ET.register_namespace("xsi", XSI_NS)

# Example: 32 deg 36' 23.66" N
DMS_RE = re.compile(
    r'^\s*(\d+(?:\.\d+)?)\s*(?:deg|°)\s*'
    r'(\d+(?:\.\d+)?)\s*[\u2032\']\s*'
    r'(\d+(?:\.\d+)?)\s*["\u2033]?\s*([NSEW])\s*$',
    re.IGNORECASE,
)

DATE_PREFIX_RE = re.compile(r"^(\d{4}):(\d{2}):(\d{2})\s+")

REQUIRED_FIELDS = ("GPS Latitude", "GPS Longitude", "GPS Date/Time")
GPS_FIELDS = set(REQUIRED_FIELDS) | {"Absolute Altitude"}


@dataclass(frozen=True)
class TrackPoint:
    latitude: float
    longitude: float
    timestamp: str  # ISO 8601, UTC
    altitude_m: float | None = None


def parse_coordinate(raw: str, *, latitude: bool) -> float:
    """Parse ExifTool's DMS coordinates, or signed decimal degrees."""
    raw = raw.strip()
    match = DMS_RE.fullmatch(raw)

    if match:
        deg, minutes, seconds = (float(n) for n in match.group(1, 2, 3))
        hemisphere = match.group(4).upper()
        if not (0 <= minutes < 60 and 0 <= seconds < 60):
            raise ValueError(f"invalid DMS components: {raw!r}")
        if hemisphere not in ("NS" if latitude else "EW"):
            raise ValueError(f"wrong hemisphere for coordinate: {raw!r}")
        value = deg + minutes / 60 + seconds / 3600
        if hemisphere in ("S", "W"):
            value = -value
    else:
        # Also accept decimal degrees such as -111.245875 or 32.60657 N.
        decimal = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*([NSEW])?", raw, re.I)
        if not decimal:
            raise ValueError(f"unrecognized GPS coordinate: {raw!r}")
        value = float(decimal.group(1))
        hemisphere = decimal.group(2)
        if hemisphere:
            hemisphere = hemisphere.upper()
            if hemisphere not in ("NS" if latitude else "EW"):
                raise ValueError(f"wrong hemisphere for coordinate: {raw!r}")
            if value < 0 and hemisphere in ("N", "E"):
                raise ValueError(f"conflicting sign and hemisphere: {raw!r}")
            value = abs(value) * (-1 if hemisphere in ("S", "W") else 1)

    maximum = 90 if latitude else 180
    if not math.isfinite(value) or not -maximum <= value <= maximum:
        raise ValueError(f"GPS coordinate out of range: {raw!r}")
    return value


def parse_timestamp(raw: str) -> str:
    """Convert ExifTool GPS Date/Time to an ISO-8601 UTC timestamp for GPX."""
    # ExifTool: 2026:05:09 17:57:47.033Z
    # ISO 8601: 2026-05-09T17:57:47.033Z
    iso = DATE_PREFIX_RE.sub(r"\1-\2-\3T", raw.strip(), count=1)
    iso = iso.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError as exc:
        raise ValueError(f"invalid GPS Date/Time: {raw!r}") from exc

    # GPS Date/Time is defined as UTC; only apply this default when no zone is given.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)

    fraction = re.search(r"\.(\d+)", raw)
    if fraction and len(fraction.group(1)) > 3:
        timespec = "microseconds"
    elif fraction:
        timespec = "milliseconds"
    else:
        timespec = "seconds"
    return dt.isoformat(timespec=timespec).replace("+00:00", "Z")


def parse_sample(sample: dict[str, str]) -> TrackPoint:
    missing = [key for key in REQUIRED_FIELDS if key not in sample]
    if missing:
        raise ValueError("missing " + ", ".join(missing))

    lat = parse_coordinate(sample["GPS Latitude"], latitude=True)
    lon = parse_coordinate(sample["GPS Longitude"], latitude=False)
    timestamp = parse_timestamp(sample["GPS Date/Time"])

    altitude = None
    if "Absolute Altitude" in sample:
        altitude = float(sample["Absolute Altitude"].removesuffix(" m"))
        if not math.isfinite(altitude):
            raise ValueError("non-finite Absolute Altitude")

    return TrackPoint(lat, lon, timestamp, altitude)


def read_samples(path: Path) -> tuple[list[TrackPoint], int, list[str]]:
    """Group lines by Sample Time, avoiding fields from other MP4 tracks.

    Sample Time begins each repeated DJI telemetry record; GPS Date/Time
    is near its end. The Sample Time value itself may be rounded by ExifTool,
    so we use GPS Date/Time for the actual GPX timestamp.
    """
    points: list[TrackPoint] = []
    total_samples = 0
    problems: list[str] = []
    current: dict[str, str] | None = None
    start_line = 0

    def finish_sample() -> None:
        nonlocal total_samples
        if current is None:
            return
        total_samples += 1
        try:
            points.append(parse_sample(current))
        except (ValueError, OverflowError) as exc:
            problems.append(f"sample starting at line {start_line}: {exc}")

    with path.open("r", encoding="utf-8-sig", errors="replace") as file:
        for line_number, line in enumerate(file, 1):
            key, sep, value = line.partition(":")
            if not sep:
                continue
            key, value = key.strip(), value.strip()

            if key == "Sample Time":
                finish_sample()
                current = {}
                start_line = line_number
            elif key == "Track Header Version" and current is not None:
                # Previous telemetry track ended; remaining tags may belong
                # to unrelated MP4 tracks (e.g. DJI dbgi metadata).
                finish_sample()
                current = None
            elif current is not None and key in GPS_FIELDS:
                current[key] = value

    finish_sample()
    return points, total_samples, problems


def make_gpx(points: list[TrackPoint], track_name: str) -> ET.ElementTree:
    root = ET.Element(
        f"{{{GPX_NS}}}gpx",
        {
            "version": "1.1",
            "creator": "dji_exiftool_to_gpx.py",
            f"{{{XSI_NS}}}schemaLocation": f"{GPX_NS} {SCHEMA_URL}",
        },
    )

    metadata = ET.SubElement(root, f"{{{GPX_NS}}}metadata")
    ET.SubElement(metadata, f"{{{GPX_NS}}}name").text = track_name

    track = ET.SubElement(root, f"{{{GPX_NS}}}trk")
    ET.SubElement(track, f"{{{GPX_NS}}}name").text = track_name
    segment = ET.SubElement(track, f"{{{GPX_NS}}}trkseg")

    for point in points:
        trkpt = ET.SubElement(
            segment,
            f"{{{GPX_NS}}}trkpt",
            {"lat": f"{point.latitude:.8f}", "lon": f"{point.longitude:.8f}"},
        )
        if point.altitude_m is not None:
            ET.SubElement(trkpt, f"{{{GPX_NS}}}ele").text = f"{point.altitude_m:.3f}"
        ET.SubElement(trkpt, f"{{{GPX_NS}}}time").text = point.timestamp

    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert the text output of 'exiftool -ee VIDEO.MP4' to a GPX track."
    )
    parser.add_argument("-i", "--input", required=True, type=Path, help="ExifTool plain-text output (.txt)")
    parser.add_argument("-o", "--output", type=Path, help="GPX output path (default: input name with .gpx)")
    parser.add_argument("--overwrite", action="store_true", help="Permit replacing an existing output GPX")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.with_suffix(".gpx")

    if not input_path.is_file():
        parser.error(f"input file does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        parser.error("input and output must be different paths")
    if output_path.exists() and not args.overwrite:
        parser.error(f"output already exists: {output_path} (use --overwrite to replace it)")

    try:
        points, total, problems = read_samples(input_path)
        if not points:
            print("[ERROR] No valid GPS samples found in input file.", file=sys.stderr)
            for problem in problems[:5]:
                print(f"[WARNING] {problem}", file=sys.stderr)
            return 1

        for problem in problems[:5]:
            print(f"[WARNING] Skipped {problem}", file=sys.stderr)
        if len(problems) > 5:
            print(f"[WARNING] ... and {len(problems) - 5} additional invalid samples.", file=sys.stderr)

        # Keep ExifTool's sample order. A GPX track is an ordered sequence,
        # and repeated coordinates with distinct timestamps are intentional.
        tree = make_gpx(points, input_path.stem)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    except (OSError, UnicodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"Input:        {input_path}")
    print(f"Output:       {output_path.resolve()}")
    print(f"Samples read: {total}")
    print(f"Track points: {len(points)}")
    print(f"Skipped:      {total - len(points)}")
    print(f"First UTC:    {points[0].timestamp}")
    print(f"Last UTC:     {points[-1].timestamp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
