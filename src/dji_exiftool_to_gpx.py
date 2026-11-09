#!/usr/bin/env python3
"""Convert DJI ExifTool -ee text output into a GPX track and time-series plots.

Usage:
    python3 dji_exiftool_to_gpx.py DJI_video.txt
    python3 dji_exiftool_to_gpx.py DJI_video.txt -o flight.gpx --plot-output exposure.png
    python3 dji_exiftool_to_gpx.py DJI_video.txt --attitude-output attitude.png
    python3 dji_exiftool_to_gpx.py DJI_video.txt --unwrap-yaw
    python3 dji_exiftool_to_gpx.py DJI_video.txt --no-plot

Input is *only* the text file produced by 'exiftool -ee VIDEO.MP4'.
The script never reads the MP4 file or invokes ExifTool.

Requires matplotlib for the exposure and attitude graphs:
    python3 -m pip install matplotlib
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
EXPOSURE_FIELDS = ("ISO", "Shutter Speed", "F Number")
ATTITUDE_FIELDS = ("Drone Roll", "Drone Pitch", "Drone Yaw")
SAMPLE_FIELDS = (set(REQUIRED_FIELDS) | {"Absolute Altitude"}
                 | set(EXPOSURE_FIELDS) | set(ATTITUDE_FIELDS))


@dataclass(frozen=True)
class TrackPoint:
    latitude: float
    longitude: float
    timestamp: str  # ISO 8601 UTC, preserving fraction of a second
    altitude_m: float | None = None
    iso: float | None = None
    shutter_seconds: float | None = None
    f_number: float | None = None
    roll_deg: float | None = None
    pitch_deg: float | None = None
    yaw_deg: float | None = None


def parse_coordinate(raw: str, *, latitude: bool) -> float:
    """Parse ExifTool DMS coordinates or signed decimal degrees."""
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
    iso = DATE_PREFIX_RE.sub(r"\1-\2-\3T", raw.strip(), count=1)
    iso = iso.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError as exc:
        raise ValueError(f"invalid GPS Date/Time: {raw!r}") from exc

    # ExifTool's GPS Date/Time is normally UTC; default to UTC if unspecified.
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


def parse_positive_number(raw: str | None) -> float | None:
    """Return finite positive telemetry values, or None if absent/bad."""
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


def parse_signed_angle(raw: str | None) -> float | None:
    """Parse a signed angle in degrees; zero and negative values are valid."""
    if raw is None:
        return None
    try:
        value = float(raw.strip().removesuffix("°").strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_shutter_seconds(raw: str | None) -> float | None:
    """Parse ExifTool shutter speeds such as '1/100', '1/131', or '0.01'."""
    if raw is None:
        return None
    # Fractions avoid any dependence on locale and support numeric strings too.
    raw = re.sub(r"\s*(?:s|sec|seconds)$", "", raw.strip(), flags=re.I)
    try:
        value = float(Fraction(raw))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return value if math.isfinite(value) and value > 0 else None


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

    # Missing or malformed exposure data does NOT invalidate a GPS sample.
    # The graph uses NaN for missing readings and retains every timestamp.
    return TrackPoint(
        latitude=lat,
        longitude=lon,
        timestamp=timestamp,
        altitude_m=altitude,
        iso=parse_positive_number(sample.get("ISO")),
        shutter_seconds=parse_shutter_seconds(sample.get("Shutter Speed")),
        f_number=parse_positive_number(sample.get("F Number")),
        roll_deg=parse_signed_angle(sample.get("Drone Roll")),
        pitch_deg=parse_signed_angle(sample.get("Drone Pitch")),
        yaw_deg=parse_signed_angle(sample.get("Drone Yaw")),
    )


def read_samples(path: Path) -> tuple[list[TrackPoint], int, list[str]]:
    """Group repeated DJI telemetry records by Sample Time, in file order.

    Each Sample Time starts a new record. Its displayed value may be rounded,
    so never use that number as the graph or GPX timestamp; use GPS Date/Time.
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
                # A subsequent MP4 metadata track is not a DJI telemetry sample.
                finish_sample()
                current = None
            elif current is not None and key in SAMPLE_FIELDS:
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


def make_exposure_plot(points: list[TrackPoint], track_name: str, output: Path) -> None:
    """Draw all GPS-aligned exposure samples on three shared-time axes.

    Every data series has exactly len(points) entries. A missing exposure
    value is represented by NaN, so it never shifts later values in time.
    """
    import matplotlib

    matplotlib.use("Agg")  # Works on headless Ubuntu servers.
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # Use true per-sample UTC timestamps, not rounded Sample Time strings.
    times = [datetime.fromisoformat(p.timestamp.replace("Z", "+00:00")) for p in points]
    iso = [p.iso if p.iso is not None else math.nan for p in points]
    shutter_seconds = [
        p.shutter_seconds if p.shutter_seconds is not None else math.nan
        for p in points
    ]
    f_number = [p.f_number if p.f_number is not None else math.nan for p in points]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"DJI exposure triangle — {track_name}\n"
        f"{times[0]:%Y-%m-%d} UTC  |  {len(points):,} timestamped samples",
        fontsize=15,
    )

    # Use distinct colors to make each exposure-triangle parameter easy to identify.
    for ax, values, color in zip(
        axes,
        (iso, shutter_seconds, f_number),
        ("tab:blue", "tab:orange", "tab:green"),
    ):
        # A marker is plotted for each reading; the connecting line shows trends.
        ax.plot(times, values, color=color, linewidth=0.8, marker=".", markersize=1.6)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.01)

    axes[0].set_ylabel("ISO")
    axes[0].set_title("Sensor sensitivity (ISO)", loc="left", fontsize=11)

    # Plot exposure duration directly in seconds, with decimal tick labels.
    # Four decimal places are enough to display shutter times near 1/100 s.
    def shutter_tick(seconds: float, _: int) -> str:
        return f"{seconds:.4f}".rstrip("0").rstrip(".")

    axes[1].set_ylabel("Exposure time (s)")
    axes[1].set_title("Exposure time (shutter speed)", loc="left", fontsize=11)
    axes[1].yaxis.set_major_formatter(FuncFormatter(shutter_tick))

    axes[2].set_ylabel("Aperture")
    axes[2].set_title("Lens aperture (f-number)", loc="left", fontsize=11)
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[2].set_xlabel("GPS Date/Time (UTC, HH:MM:SS)")
    axes[2].xaxis.set_major_locator(mdates.AutoDateLocator(tz=timezone.utc))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))

    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)



def unwrap_yaw_degrees(values: list[float]) -> list[float]:
    """Unwrap successive DJI yaw angles across +/-180 degrees.

    Missing readings stay as NaN; unwrapping uses the previous valid sample.
    This is a display-only transformation; the input telemetry is unmodified.
    """
    output: list[float] = []
    previous_raw: float | None = None
    previous_continuous: float | None = None
    for value in values:
        if not math.isfinite(value):
            output.append(math.nan)
            continue
        if previous_raw is None or previous_continuous is None:
            continuous = value
        else:
            # Find the shortest signed angular difference modulo 360 degrees.
            delta = (value - previous_raw + 180.0) % 360.0 - 180.0
            continuous = previous_continuous + delta
        output.append(continuous)
        previous_raw = value
        previous_continuous = continuous
    return output


def make_attitude_plot(
    points: list[TrackPoint], track_name: str, output: Path, *, unwrap_yaw: bool = False
) -> None:
    """Draw native DJI roll/pitch/yaw in degrees, sharing GPS time on x-axis.

    Every series has one slot per GPS trackpoint. Missing angles create gaps;
    no samples are invented. Raw yaw is split at +/-180-degree wraparound so
    the line does not falsely imply an abrupt 360-degree rotation.
    """
    import matplotlib

    matplotlib.use("Agg")  # Safe to run without a graphical desktop.
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    times = [datetime.fromisoformat(p.timestamp.replace("Z", "+00:00")) for p in points]
    roll = [p.roll_deg if p.roll_deg is not None else math.nan for p in points]
    pitch = [p.pitch_deg if p.pitch_deg is not None else math.nan for p in points]
    yaw_raw = [p.yaw_deg if p.yaw_deg is not None else math.nan for p in points]
    yaw = unwrap_yaw_degrees(yaw_raw) if unwrap_yaw else yaw_raw

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"DJI drone attitude — {track_name}\n"
        f"{times[0]:%Y-%m-%d} UTC  |  {len(points):,} GPS-timestamped samples",
        fontsize=15,
    )

    for ax, values, label, color in zip(
        axes,
        (roll, pitch, yaw),
        ("Roll (°)", "Pitch (°)", "Yaw (°)"),
        ("tab:blue", "tab:orange", "tab:green"),
    ):
        ax.set_ylabel(label)
        ax.set_title(label.replace(" (°)", " angle"), loc="left", fontsize=11)
        ax.axhline(0, linewidth=0.75, color="gray", alpha=0.6)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.01)
        if ax is not axes[2] or unwrap_yaw:
            ax.plot(times, values, color=color, linewidth=0.9,
                    marker=".", markersize=1.8)

    if not unwrap_yaw:
        # Preserve all raw yaw samples. The markers remain visible on both
        # sides of wrap boundaries; separate line segments avoid a false jump.
        ax = axes[2]
        ax.plot(times, yaw, linestyle="None", color="tab:green",
                marker=".", markersize=1.8)
        xs, ys = [], []
        for timestamp, angle in zip(times, yaw):
            if not math.isfinite(angle) or (ys and abs(angle - ys[-1]) > 180):
                if xs:
                    ax.plot(xs, ys, color="tab:green", linewidth=0.9)
                xs, ys = [], []
            if math.isfinite(angle):
                xs.append(timestamp)
                ys.append(angle)
        if xs:
            ax.plot(xs, ys, color="tab:green", linewidth=0.9)

    axes[2].set_title(
        "Yaw angle (continuous/unwrapped)" if unwrap_yaw else "Yaw angle (native DJI, ±180° wrap)",
        loc="left", fontsize=11,
    )
    axes[2].set_xlabel("GPS Date/Time (UTC, HH:MM:SS)")
    axes[2].xaxis.set_major_locator(mdates.AutoDateLocator(tz=timezone.utc))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))

    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Convert 'exiftool -ee VIDEO.MP4' text output into a GPX track "
                     "and timestamp-aligned exposure/attitude PNG graphs.")
    )
    parser.add_argument("input", type=Path, help="ExifTool plain-text output (.txt)")
    parser.add_argument("-o", "--output", type=Path,
                        help="GPX output path (default: input name with .gpx)")
    parser.add_argument("--plot-output", type=Path,
                        help="Exposure graph PNG (default: INPUT_exposure.png)")
    parser.add_argument("--attitude-output", type=Path,
                        help="Drone attitude PNG (default: INPUT_attitude.png)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Generate only the GPX, without either graph")
    parser.add_argument("--no-exposure", action="store_true",
                        help="Skip the exposure graph but still generate attitude graph")
    parser.add_argument("--no-attitude", action="store_true",
                        help="Skip the attitude graph but still generate exposure graph")
    parser.add_argument("--unwrap-yaw", action="store_true",
                        help="Plot continuous yaw across +/-180 degrees instead of raw yaw")
    parser.add_argument("--overwrite", action="store_true",
                        help="Permit replacing existing output files")
    args = parser.parse_args()

    input_path = args.input
    gpx_path = args.output or input_path.with_suffix(".gpx")
    plot_path = args.plot_output or input_path.with_name(input_path.stem + "_exposure.png")
    attitude_path = args.attitude_output or input_path.with_name(input_path.stem + "_attitude.png")
    do_exposure = not (args.no_plot or args.no_exposure)
    do_attitude = not (args.no_plot or args.no_attitude)

    if not input_path.is_file():
        parser.error(f"input file does not exist: {input_path}")

    paths = [gpx_path]
    if do_exposure:
        paths.append(plot_path)
    if do_attitude:
        paths.append(attitude_path)
    resolved = [p.resolve() for p in paths]
    if input_path.resolve() in resolved:
        parser.error("input and output paths must be different")
    if len(set(resolved)) != len(resolved):
        parser.error("GPX and PNG output paths must all be different")
    for path in paths:
        if path.exists() and not args.overwrite:
            parser.error(f"output already exists: {path} (use --overwrite to replace it)")
        if not path.parent.is_dir():
            parser.error(f"output directory does not exist: {path.parent}")

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
            print(f"[WARNING] ... and {len(problems) - 5} additional invalid samples.",
                  file=sys.stderr)

        if do_exposure or do_attitude:
            try:
                import matplotlib  # noqa: F401
            except ImportError:
                print("[ERROR] Plotting requires matplotlib: "
                      "python3 -m pip install matplotlib", file=sys.stderr)
                return 1

        # Preserve ExifTool's sample ordering; do not thin or deduplicate.
        make_gpx(points, input_path.stem).write(
            gpx_path, encoding="utf-8", xml_declaration=True
        )
        if do_exposure:
            make_exposure_plot(points, input_path.stem, plot_path)
        if do_attitude:
            make_attitude_plot(points, input_path.stem, attitude_path,
                               unwrap_yaw=args.unwrap_yaw)

    except (OSError, UnicodeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"Input:         {input_path}")
    print(f"GPX output:    {gpx_path.resolve()}")
    if do_exposure:
        print(f"Exposure PNG:  {plot_path.resolve()}")
    if do_attitude:
        print(f"Attitude PNG:  {attitude_path.resolve()}")
    print(f"Samples read:  {total}")
    print(f"GPX points:    {len(points)}")
    print(f"Skipped:       {total - len(points)}")
    print(f"First UTC:     {points[0].timestamp}")
    print(f"Last UTC:      {points[-1].timestamp}")
    metrics = []
    if do_exposure:
        metrics.extend((("ISO", "iso"), ("Shutter", "shutter_seconds"),
                        ("F-number", "f_number")))
    if do_attitude:
        metrics.extend((("Roll", "roll_deg"), ("Pitch", "pitch_deg"),
                        ("Yaw", "yaw_deg")))
    for label, attr in metrics:
        available = sum(getattr(point, attr) is not None for point in points)
        print(f"{label + ' points:':<15}{available} / {len(points)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
