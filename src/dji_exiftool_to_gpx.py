#!/usr/bin/env python3
"""Convert ExifTool -ee DJI MP4 text to GPX, exposure and attitude PNGs.

Requires: python3 -m pip install matplotlib
Input: ExifTool-generated .txt file, NOT the MP4.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


NS = "http://www.topografix.com/GPX/1/1"

ET.register_namespace("", NS)

FIELDS = {
    "GPS Latitude",
    "GPS Longitude",
    "GPS Date/Time",
    "Absolute Altitude",
    "ISO",
    "Shutter Speed",
    "F Number",
    "Drone Roll",
    "Drone Pitch",
    "Drone Yaw",
    "Gimbal Yaw",
}

DMS = re.compile(
    r"""(\d+(?:\.\d+)?)\s*(?:deg|°)\s*(\d+(?:\.\d+)?)\s*'\s*"""
    r"""(\d+(?:\.\d+)?)\s*"?\s*([NSEW])""",
    re.I,
)


@dataclass
class Sample:
    lat: float
    lon: float
    time: datetime
    altitude: float | None
    iso: float | None
    shutter: float | None
    aperture: float | None
    roll: float | None
    pitch: float | None
    yaw: float | None
    gimbal_yaw: float | None


def coordinate(raw, latitude):
    m = DMS.fullmatch(raw.strip())

    if m:
        deg, minutes, seconds = map(float, m.group(1, 2, 3))
        direction = m.group(4).upper()

        if not (0 <= minutes < 60 and 0 <= seconds < 60):
            raise ValueError(f"Invalid DMS coordinate: {raw}")

        result = deg + minutes / 60 + seconds / 3600
        result *= -1 if direction in "SW" else 1

    else:
        dm = re.fullmatch(
            r"([+-]?\d+(?:\.\d+)?)\s*([NSEW])?",
            raw.strip(),
            re.I,
        )

        if not dm:
            raise ValueError(f"Invalid GPS coordinate: {raw}")

        result = float(dm.group(1))
        direction = (dm.group(2) or "").upper()

        if direction:
            result = abs(result) * (-1 if direction in "SW" else 1)

    if direction and direction not in ("NS" if latitude else "EW"):
        raise ValueError(f"Wrong GPS hemisphere: {raw}")

    limit = 90 if latitude else 180

    if not math.isfinite(result) or abs(result) > limit:
        raise ValueError(f"Out-of-range GPS coordinate: {raw}")

    return result


def utc_datetime(raw):
    text = re.sub(
        r"^(\d{4}):(\d\d):(\d\d)\s+",
        r"\1-\2-\3T",
        raw.strip(),
    )

    dt = datetime.fromisoformat(
        text.replace("Z", "+00:00")
    )

    return (
        dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    ).astimezone(timezone.utc)


def number(raw, positive=False):
    if raw is None:
        return None

    try:
        value = float(
            raw.strip()
            .removesuffix("°")
            .removesuffix(" m")
            .strip()
        )

        if math.isfinite(value) and (not positive or value > 0):
            return value

    except ValueError:
        pass

    return None


def shutter_seconds(raw):
    if raw is None:
        return None

    try:
        value = float(
            Fraction(
                re.sub(
                    r"\s*(s|sec|seconds)$",
                    "",
                    raw.strip(),
                    flags=re.I,
                )
            )
        )

        return value if math.isfinite(value) and value > 0 else None

    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def parse_sample(rec):
    for key in (
        "GPS Latitude",
        "GPS Longitude",
        "GPS Date/Time",
    ):
        if key not in rec:
            raise ValueError(f"Missing {key}")

    altitude = number(rec.get("Absolute Altitude"))

    return Sample(
        lat=coordinate(rec["GPS Latitude"], True),
        lon=coordinate(rec["GPS Longitude"], False),
        time=utc_datetime(rec["GPS Date/Time"]),
        altitude=altitude,
        iso=number(rec.get("ISO"), positive=True),
        shutter=shutter_seconds(rec.get("Shutter Speed")),
        aperture=number(rec.get("F Number"), positive=True),
        roll=number(rec.get("Drone Roll")),
        pitch=number(rec.get("Drone Pitch")),
        yaw=number(rec.get("Drone Yaw")),
        gimbal_yaw=number(rec.get("Gimbal Yaw")),
    )


def load_samples(path):
    points, warnings = [], []
    current = None
    total = 0
    start_line = 0

    def flush():
        nonlocal total

        if current is None:
            return

        total += 1

        try:
            points.append(parse_sample(current))
        except ValueError as error:
            warnings.append(f"line {start_line}: {error}")

    with path.open(
        encoding="utf-8-sig",
        errors="replace",
    ) as stream:

        for line_no, line in enumerate(stream, 1):
            key, sep, value = line.partition(":")

            if not sep:
                continue

            key, value = key.strip(), value.strip()

            if key == "Sample Time":
                flush()
                current, start_line = {}, line_no

            elif key == "Track Header Version" and current is not None:
                flush()
                current = None

            elif current is not None and key in FIELDS:
                current[key] = value

    flush()

    return points, total, warnings


def write_gpx(points, name, output):
    root = ET.Element(
        f"{{{NS}}}gpx",
        version="1.1",
        creator="dji_exiftool_to_gpx.py",
    )

    metadata = ET.SubElement(root, f"{{{NS}}}metadata")
    ET.SubElement(metadata, f"{{{NS}}}name").text = name

    track = ET.SubElement(root, f"{{{NS}}}trk")
    ET.SubElement(track, f"{{{NS}}}name").text = name

    segment = ET.SubElement(track, f"{{{NS}}}trkseg")

    for p in points:
        pt = ET.SubElement(
            segment,
            f"{{{NS}}}trkpt",
            lat=f"{p.lat:.8f}",
            lon=f"{p.lon:.8f}",
        )

        if p.altitude is not None:
            ET.SubElement(
                pt,
                f"{{{NS}}}ele",
            ).text = f"{p.altitude:.3f}"

        ET.SubElement(
            pt,
            f"{{{NS}}}time",
        ).text = (
            p.time.isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    ET.indent(root, space="  ")

    ET.ElementTree(root).write(
        output,
        encoding="utf-8",
        xml_declaration=True,
    )


def setup_time_axis(axis):
    import matplotlib.dates as mdates

    axis.set_xlabel("GPS Date/Time (UTC)")

    axis.xaxis.set_major_locator(
        mdates.AutoDateLocator(tz=timezone.utc)
    )

    axis.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%H:%M:%S",
            tz=timezone.utc,
        )
    )


def plot_exposure(points, name, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    times = [p.time for p in points]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True,
        constrained_layout=True,
    )

    fig.suptitle(
        f"DJI exposure triangle — {name} "
        f"({len(points):,} samples)"
    )

    specs = [
        ("iso", "ISO", "tab:blue"),
        ("shutter", "Exposure time (s)", "tab:orange"),
        ("aperture", "F-number", "tab:green"),
    ]

    for ax, (field, label, color) in zip(axes, specs):

        readings = [
            getattr(p, field)
            if getattr(p, field) is not None
            else math.nan
            for p in points
        ]

        ax.plot(
            times,
            readings,
            ".",
            color=color,
            markersize=1.6,
        )

        ax.plot(
            times,
            readings,
            color=color,
            linewidth=0.8,
        )

        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    # Decimal exposure times, e.g. 0.01 instead of 1/100.
    axes[1].yaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _: f"{v:.4f}".rstrip("0").rstrip(".")
        )
    )

    # Numeric F-numbers without the f/ prefix.
    axes[2].yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:g}")
    )

    setup_time_axis(axes[-1])

    fig.savefig(
        output,
        dpi=160,
        facecolor="white",
    )

    plt.close(fig)


def unwrap_angles(values):
    result = []
    previous_raw = None
    previous_unwrapped = None

    for angle in values:

        if not math.isfinite(angle):
            result.append(math.nan)
            continue

        if previous_raw is None:
            unwrapped = angle

        else:
            delta = (
                (angle - previous_raw + 180) % 360
            ) - 180

            unwrapped = previous_unwrapped + delta

        result.append(unwrapped)

        previous_raw = angle
        previous_unwrapped = unwrapped

    return result


def plot_attitude(points, name, output, unwrap_yaw=False):
    import matplotlib.pyplot as plt

    times = [p.time for p in points]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 11),
        sharex=True,
        constrained_layout=True,
    )

    fig.suptitle(
        f"DJI drone and gimbal attitude — {name} "
        f"({len(points):,} samples)"
    )

    specs = [
        ("roll", "Drone roll (°)", "tab:blue"),
        ("pitch", "Drone pitch (°)", "tab:orange"),
        ("yaw", "Drone yaw (°)", "tab:green"),
        ("gimbal_yaw", "Gimbal yaw (°)", "tab:purple"),
    ]

    for ax, (field, label, color) in zip(axes, specs):

        raw = [
            getattr(p, field)
            if getattr(p, field) is not None
            else math.nan
            for p in points
        ]

        if unwrap_yaw and field in ("yaw", "gimbal_yaw"):
            values = unwrap_angles(raw)
        else:
            values = raw

        ax.plot(
            times,
            values,
            ".",
            color=color,
            markersize=1.8,
        )

        # Avoid false connecting lines across the +/-180° boundary.
        if field in ("yaw", "gimbal_yaw") and not unwrap_yaw:

            values_for_line = values[:]

            for i in range(1, len(values_for_line)):

                if (
                    math.isfinite(raw[i])
                    and math.isfinite(raw[i - 1])
                    and abs(raw[i] - raw[i - 1]) > 180
                ):
                    values_for_line[i] = math.nan

            ax.plot(
                times,
                values_for_line,
                color=color,
                linewidth=0.9,
            )

        else:
            ax.plot(
                times,
                values,
                color=color,
                linewidth=0.9,
            )

        ax.set_ylabel(label)

        ax.axhline(
            0,
            color="gray",
            alpha=0.6,
            linewidth=0.75,
        )

        ax.grid(alpha=0.3)

    setup_time_axis(axes[-1])

    fig.savefig(
        output,
        dpi=160,
        facecolor="white",
    )

    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__
    )

    p.add_argument(
        "input",
        type=Path,
        help="ExifTool .txt input (not the video)",
    )

    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="GPX output path",
    )

    p.add_argument(
        "--plot-output",
        type=Path,
        help="Exposure PNG output path",
    )

    p.add_argument(
        "--attitude-output",
        type=Path,
        help="Attitude PNG output path",
    )

    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip both graphs",
    )

    p.add_argument(
        "--no-exposure",
        action="store_true",
        help="Skip exposure graph",
    )

    p.add_argument(
        "--no-attitude",
        action="store_true",
        help="Skip attitude graph",
    )

    p.add_argument(
        "--unwrap-yaw",
        action="store_true",
        help="Unwrap drone and gimbal yaw",
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing output files",
    )

    args = p.parse_args()

    if not args.input.is_file():
        p.error(f"Input file not found: {args.input}")

    gpx = args.output or args.input.with_suffix(".gpx")

    exposure = args.plot_output or args.input.with_name(
        args.input.stem + "_exposure.png"
    )

    attitude = args.attitude_output or args.input.with_name(
        args.input.stem + "_attitude.png"
    )

    do_exposure = not (args.no_plot or args.no_exposure)
    do_attitude = not (args.no_plot or args.no_attitude)

    outputs = (
        [gpx]
        + ([exposure] if do_exposure else [])
        + ([attitude] if do_attitude else [])
    )

    all_paths = outputs + [args.input]

    if (
        len(set(path.resolve() for path in all_paths))
        != len(all_paths)
    ):
        p.error("Input and output filenames must all differ")

    for path in outputs:

        if path.exists() and not args.overwrite:
            p.error(
                f"Output exists: {path}; use --overwrite"
            )

        if not path.parent.is_dir():
            p.error(
                f"Output directory does not exist: {path.parent}"
            )

    points, total, warnings = load_samples(args.input)

    if not points:
        p.error("No GPS samples found")

    for message in warnings[:5]:
        print(
            f"[WARNING] {message}",
            file=sys.stderr,
        )

    if len(warnings) > 5:
        print(
            f"[WARNING] {len(warnings) - 5} more invalid records",
            file=sys.stderr,
        )

    if do_exposure or do_attitude:
        try:
            import matplotlib
        except ImportError:
            p.error(
                "Install matplotlib: "
                "python3 -m pip install matplotlib"
            )

        matplotlib.use("Agg")

    write_gpx(
        points,
        args.input.stem,
        gpx,
    )

    if do_exposure:
        plot_exposure(
            points,
            args.input.stem,
            exposure,
        )

    if do_attitude:
        plot_attitude(
            points,
            args.input.stem,
            attitude,
            args.unwrap_yaw,
        )

    print(
        f"GPS samples: {len(points)} / {total}\n"
        f"GPX: {gpx.resolve()}"
    )

    if do_exposure:
        print(f"Exposure PNG: {exposure.resolve()}")

    if do_attitude:
        print(f"Attitude PNG: {attitude.resolve()}")

    for field in (
        "iso",
        "shutter",
        "aperture",
        "roll",
        "pitch",
        "yaw",
        "gimbal_yaw",
    ):
        count = sum(
            getattr(point, field) is not None
            for point in points
        )

        print(
            f"{field}: {count} / {len(points)}"
        )


if __name__ == "__main__":
    main()