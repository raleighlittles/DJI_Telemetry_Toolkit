#!/usr/bin/env python3
"""Convert DJI ExifTool -ee text output into a GPX track and time-series plots.

Usage:
    python3 dji_exiftool_to_gpx.py DJI_video.txt
    python3 dji_exiftool_to_gpx.py DJI_video.txt -o flight.gpx --plot-output exposure.png
    python3 dji_exiftool_to_gpx.py DJI_video.txt --flight_telemetry-output flight_telemetry.png
    python3 dji_exiftool_to_gpx.py DJI_video.txt --unwrap-yaw
    python3 dji_exiftool_to_gpx.py DJI_video.txt --no-plot

Input is *only* the text file produced by 'exiftool -ee VIDEO.MP4'.
The script never reads the MP4 file or invokes ExifTool.

Requires matplotlib for the exposure and flight_telemetry graphs:
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
DJI_EXT_NS = "urn:dji-exiftool:gpx-extensions:1"  # Custom GPX metadata extensions

ET.register_namespace("", GPX_NS)
ET.register_namespace("xsi", XSI_NS)
ET.register_namespace("dji", DJI_EXT_NS)

# Example: 32 deg 36' 23.66" N
DMS_RE = re.compile(
    r'^\s*(\d+(?:\.\d+)?)\s*(?:deg|°)\s*'
    r'(\d+(?:\.\d+)?)\s*[\u2032\']\s*'
    r'(\d+(?:\.\d+)?)\s*["\u2033]?\s*([NSEW])\s*$',
    re.IGNORECASE,
)
DATE_PREFIX_RE = re.compile(r"^(\d{4}):(\d{2}):(\d{2})\s+")

REQUIRED_FIELDS = ("GPS Latitude", "GPS Longitude", "GPS Date/Time")
EXPOSURE_FIELDS = ("ISO", "Shutter Speed", "F Number", "Color Temperature")
ATTITUDE_FIELDS = ("Drone Roll", "Drone Pitch", "Drone Yaw", "Gimbal Yaw")
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
    color_temperature_k: float | None = None
    roll_deg: float | None = None
    pitch_deg: float | None = None
    yaw_deg: float | None = None
    gimbal_yaw_deg: float | None = None


@dataclass(frozen=True)
class VideoMetadata:
    """Video-level ExifTool fields, not repeated per GPS sample."""
    resolution: tuple[int, int] | None = None
    model: str | None = None
    serial_number: str | None = None


def video_subtitle(
    metadata: VideoMetadata,
    first_timestamp: datetime,
    sample_count: int,
    *,
    include_resolution: bool = False,
) -> str:
    """Shared, consistently formatted subtitles for both figures."""
    model = metadata.model or "Unavailable"
    serial = metadata.serial_number or "Unavailable"
    identity_line = f"Model: {model}  |  Serial Number: {serial}"

    details = []
    if include_resolution:
        resolution = metadata.resolution
        details.append(
            f"Resolution: {resolution[0]} × {resolution[1]}"
            if resolution else "Resolution unavailable"
        )
    details.extend((
        f"{first_timestamp:%Y-%m-%d} UTC",
        f"{sample_count:,} timestamped samples",
    ))
    return identity_line + "\n" + "  |  ".join(details)


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
        value = float(raw.strip())
    except (ValueError, TypeError):
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
        color_temperature_k=parse_positive_number(sample.get("Color Temperature")),
        roll_deg=parse_signed_angle(sample.get("Drone Roll")),
        pitch_deg=parse_signed_angle(sample.get("Drone Pitch")),
        yaw_deg=parse_signed_angle(sample.get("Drone Yaw")),
        gimbal_yaw_deg=parse_signed_angle(sample.get("Gimbal Yaw")),
    )


def read_samples(path: Path) -> tuple[list[TrackPoint], int, list[str], VideoMetadata]:
    """Group repeated DJI telemetry records by Sample Time, in file order.

    Each Sample Time starts a new record. Its displayed value may be rounded,
    so never use that number as the graph or GPX timestamp; use GPS Date/Time.
    """
    points: list[TrackPoint] = []
    total_samples = 0
    problems: list[str] = []
    current: dict[str, str] | None = None
    start_line = 0
    dimensions: dict[str, int] = {}
    identity: dict[str, str] = {}

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

            # Frame Width/Height are video-level metadata, before the samples.
            # Capture the first valid occurrence of each, not from GPS readings.
            if key in ("Frame Width", "Frame Height") and key not in dimensions:
                if re.fullmatch(r"\d+", value) and int(value) > 0:
                    dimensions[key] = int(value)

            # These DJI tags are encountered within the first Sample Time block
            # in this file, so read them whether or not a sample is active.
            if key in ("Model", "Serial Number") and key not in identity and value:
                identity[key] = value

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
    width = dimensions.get("Frame Width")
    height = dimensions.get("Frame Height")
    resolution = (width, height) if width is not None and height is not None else None
    metadata = VideoMetadata(
        resolution=resolution,
        model=identity.get("Model"),
        serial_number=identity.get("Serial Number"),
    )
    return points, total_samples, problems, metadata


def make_gpx(
    points: list[TrackPoint], track_name: str, video_metadata: VideoMetadata
) -> ET.ElementTree:
    """Build GPX 1.1 with device identity in metadata description and extensions."""
    root = ET.Element(
        f"{{{GPX_NS}}}gpx",
        {
            "version": "1.1",
            "creator": "dji_exiftool_to_gpx.py",
            f"{{{XSI_NS}}}schemaLocation": f"{GPX_NS} {SCHEMA_URL}",
        },
    )

    gpx_metadata = ET.SubElement(root, f"{{{GPX_NS}}}metadata")
    ET.SubElement(gpx_metadata, f"{{{GPX_NS}}}name").text = track_name

    # Human-readable description for generic GPX viewers.
    details = []
    if video_metadata.model:
        details.append(f"Model: {video_metadata.model}")
    if video_metadata.serial_number:
        details.append(f"Serial Number: {video_metadata.serial_number}")
    if details:
        ET.SubElement(gpx_metadata, f"{{{GPX_NS}}}desc").text = " | ".join(details)

        # Structured device fields for software that understands GPX extensions.
        extensions = ET.SubElement(gpx_metadata, f"{{{GPX_NS}}}extensions")
        if video_metadata.model:
            ET.SubElement(extensions, f"{{{DJI_EXT_NS}}}Model").text = video_metadata.model
        if video_metadata.serial_number:
            ET.SubElement(extensions, f"{{{DJI_EXT_NS}}}SerialNumber").text = video_metadata.serial_number

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


def make_exposure_plot(
    points: list[TrackPoint], track_name: str, output: Path,
    metadata: VideoMetadata,
) -> None:
    """Plot ISO, shutter time, f-number, and color temperature as discrete samples.

    Each series is aligned to all GPS timestamps. Missing readings stay as NaN;
    points are not connected because camera settings may change discretely.
    """
    import matplotlib

    matplotlib.use("Agg")  # Safe on headless Ubuntu servers.
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # Use actual GPS Date/Time values rather than rounded Sample Time values.
    times = [datetime.fromisoformat(p.timestamp.replace("Z", "+00:00")) for p in points]

    # One color per variable, with four vertically aligned Cartesian plots.
    series = (
        ("iso", "ISO", "ISO (sensor sensitivity)", "tab:blue"),
        ("shutter_seconds", "Exposure time (s)", "Exposure time (seconds)", "tab:orange"),
        ("f_number", "F-number", "Lens aperture (f-number)", "tab:green"),
        ("color_temperature_k", "Color temperature (K)", "Color temperature (Kelvin)", "tab:purple"),
    )

    # Fail loudly if the fourth field was omitted from parsing. Without this
    # check Matplotlib silently renders an empty axis with nonsensical 0/-0 ticks.
    temperatures = [
        p.color_temperature_k for p in points
        if p.color_temperature_k is not None and math.isfinite(p.color_temperature_k)
    ]
    if not temperatures:
        raise ValueError(
            "No Color Temperature readings were parsed. Confirm that "
            "'Color Temperature' is in SAMPLE_FIELDS and that parse_sample() "
            "assigns color_temperature_k from sample['Color Temperature']."
        )
    print(
        f"[INFO] Color temperature: {len(temperatures):,}/{len(points):,} "
        f"samples, {min(temperatures):g}-{max(temperatures):g} K"
    )

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 11), sharex=True, constrained_layout=True
    )
    fig.suptitle(
        f"DJI exposure triangle + color temperature — {track_name}\n"
        + video_subtitle(metadata, times[0], len(points), include_resolution=True),
        fontsize=14,
    )

    for ax, (field, ylabel, title, color) in zip(axes, series):
        values = [
            getattr(point, field) if getattr(point, field) is not None else math.nan
            for point in points
        ]
        # Draw one dot per observed sample. No connecting lines or interpolation.
        ax.scatter(times, values, marker="o", s=2.5, color=color,
                   alpha=0.9, linewidths=0, rasterized=True)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.01)

    # Shutter speeds as decimal seconds (e.g., 0.01, not 1/100).
    axes[1].yaxis.set_major_formatter(
        FuncFormatter(lambda seconds, _: f"{seconds:.4f}".rstrip("0").rstrip("."))
    )
    # F-number as a raw numeric value (e.g., 1.7, not f/1.7).
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    # Color temperature in Kelvin with conventional integer ticks.
    axes[3].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
    # Do not let an empty/near-empty axis silently display ±0 Kelvin.
    if min(temperatures) == max(temperatures):
        axes[3].set_ylim(min(temperatures) - 1, max(temperatures) + 1)

    axes[-1].set_xlabel("GPS Date/Time (UTC, HH:MM:SS)")
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(tz=timezone.utc))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))

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


def make_flight_telemetry_plot(
    points: list[TrackPoint], track_name: str, output: Path,
    metadata: VideoMetadata, *, unwrap_yaw: bool = False
) -> None:
    """Draw drone and gimbal angles, plus absolute altitude, versus GPS time.

    Every series has one slot per GPS trackpoint. Missing angles create gaps;
    no samples are invented. Raw drone and gimbal yaw are split at +/-180-degree
    wraparound so a crossing does not look like an abrupt 360-degree rotation.
    """
    import matplotlib

    matplotlib.use("Agg")  # Safe to run without a graphical desktop.
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    times = [datetime.fromisoformat(p.timestamp.replace("Z", "+00:00")) for p in points]
    roll = [p.roll_deg if p.roll_deg is not None else math.nan for p in points]
    pitch = [p.pitch_deg if p.pitch_deg is not None else math.nan for p in points]
    yaw_raw = [p.yaw_deg if p.yaw_deg is not None else math.nan for p in points]
    gimbal_yaw_raw = [
        p.gimbal_yaw_deg if p.gimbal_yaw_deg is not None else math.nan
        for p in points
    ]
    yaw = unwrap_yaw_degrees(yaw_raw) if unwrap_yaw else yaw_raw
    gimbal_yaw = unwrap_yaw_degrees(gimbal_yaw_raw) if unwrap_yaw else gimbal_yaw_raw
    altitudes = [p.altitude_m if p.altitude_m is not None else math.nan for p in points]
    has_altitude = any(math.isfinite(v) for v in altitudes)
    if not has_altitude:
        print("[WARNING] No Absolute Altitude readings found; "
              "altitude panel will show no data.", file=sys.stderr)

    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"DJI flight telemetry — {track_name}\n"
        + video_subtitle(metadata, times[0], len(points)),
        fontsize=14,
    )

    for ax, values, label, color, is_yaw in zip(
        axes,
        (roll, pitch, yaw, gimbal_yaw, altitudes),
        ("Drone roll (°)", "Drone pitch (°)", "Drone yaw (°)",
         "Gimbal yaw (°)", "Absolute altitude (m)"),
        ("tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"),
        (False, False, True, True, False),
    ):
        ax.set_ylabel(label)
        ax.set_title(label, loc="left", fontsize=11)
        if label == "Absolute altitude (m)" and not has_altitude:
            ax.text(0.5, 0.5, "No Absolute Altitude data available",
                    transform=ax.transAxes, ha="center", va="center")
        if label != "Absolute altitude (m)":
            ax.axhline(0, linewidth=0.75, color="gray", alpha=0.6)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.01)
        if not is_yaw or unwrap_yaw:
            ax.plot(times, values, color=color, linewidth=0.9,
                    marker=".", markersize=1.8)
        else:
            # Plot every raw reading, but omit a connecting segment when the
            # value jumps across the +/-180 degree boundary (or is missing).
            ax.plot(times, values, linestyle="None", color=color,
                    marker=".", markersize=1.8)
            xs: list[datetime] = []
            ys: list[float] = []
            for timestamp, angle in zip(times, values):
                if not math.isfinite(angle) or (ys and abs(angle - ys[-1]) > 180):
                    if xs:
                        ax.plot(xs, ys, color=color, linewidth=0.9)
                    xs, ys = [], []
                if math.isfinite(angle):
                    xs.append(timestamp)
                    ys.append(angle)
            if xs:
                ax.plot(xs, ys, color=color, linewidth=0.9)

    if unwrap_yaw:
        axes[2].set_title("Drone yaw (continuous/unwrapped)", loc="left", fontsize=11)
        axes[3].set_title("Gimbal yaw (continuous/unwrapped)", loc="left", fontsize=11)
    else:
        axes[2].set_title("Drone yaw (native DJI, ±180° wrap)", loc="left", fontsize=11)
        axes[3].set_title("Gimbal yaw (native DJI, ±180° wrap)", loc="left", fontsize=11)

    axes[-1].set_xlabel("GPS Date/Time (UTC, HH:MM:SS)")
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(tz=timezone.utc))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))

    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Convert 'exiftool -ee VIDEO.MP4' text output into a GPX track "
                     "and timestamp-aligned exposure/flight telemetry PNG graphs.")
    )
    parser.add_argument("input", type=Path, help="ExifTool plain-text output (.txt)")
    parser.add_argument("-o", "--output", type=Path,
                        help="GPX output path (default: input name with .gpx)")
    parser.add_argument("--plot-output", type=Path,
                        help="Exposure graph PNG (default: INPUT_exposure.png)")
    parser.add_argument("--flight_telemetry-output", "--flight-telemetry-output", "--telemetry-output", "--attitude-output",
                        type=Path, dest="flight_telemetry_output",
                        help="Drone/gimbal angles + altitude PNG (default: INPUT_telemetry.png)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Generate only the GPX, without either graph")
    parser.add_argument("--no-exposure", action="store_true",
                        help="Skip exposure PNG but still generate flight telemetry PNG")
    parser.add_argument("--no-flight_telemetry", "--no-telemetry", "--no-attitude",
                        dest="no_flight_telemetry", action="store_true",
                        help="Skip flight telemetry PNG but still generate exposure PNG")
    parser.add_argument("--unwrap-yaw", action="store_true",
                        help="Plot continuous drone and gimbal yaw across +/-180 degrees")
    parser.add_argument("--overwrite", action="store_true",
                        help="Permit replacing existing output files")
    args = parser.parse_args()

    input_path = args.input
    gpx_path = args.output or input_path.with_suffix(".gpx")
    plot_path = args.plot_output or input_path.with_name(input_path.stem + "_exposure.png")
    flight_telemetry_path = args.flight_telemetry_output or input_path.with_name(input_path.stem + "_telemetry.png")
    do_exposure = not (args.no_plot or args.no_exposure)
    do_flight_telemetry = not (args.no_plot or args.no_flight_telemetry)

    if not input_path.is_file():
        parser.error(f"input file does not exist: {input_path}")

    paths = [gpx_path]
    if do_exposure:
        paths.append(plot_path)
    if do_flight_telemetry:
        paths.append(flight_telemetry_path)
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
        points, total, problems, metadata = read_samples(input_path)
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

        if do_exposure or do_flight_telemetry:
            try:
                import matplotlib  # noqa: F401
            except ImportError:
                print("[ERROR] Plotting requires matplotlib: "
                      "python3 -m pip install matplotlib", file=sys.stderr)
                return 1

        # Preserve ExifTool's sample ordering; do not thin or deduplicate.
        make_gpx(points, input_path.stem, metadata).write(
            gpx_path, encoding="utf-8", xml_declaration=True
        )
        if do_exposure:
            make_exposure_plot(points, input_path.stem, plot_path, metadata)
        if do_flight_telemetry:
            make_flight_telemetry_plot(points, input_path.stem, flight_telemetry_path,
                               metadata, unwrap_yaw=args.unwrap_yaw)

    except (OSError, UnicodeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"Input:         {input_path}")
    if metadata.resolution is not None:
        print(f"Resolution:    {metadata.resolution[0]}x{metadata.resolution[1]}")
    else:
        print("Resolution:    unavailable (Frame Width/Frame Height missing)")
    print(f"Model:         {metadata.model or 'Unavailable'}")
    print(f"Serial number: {metadata.serial_number or 'Unavailable'}")
    print(f"GPX output:    {gpx_path.resolve()}")
    if do_exposure:
        print(f"Exposure PNG:  {plot_path.resolve()}")
    if do_flight_telemetry:
        print(f"Telemetry PNG:  {flight_telemetry_path.resolve()}")
    print(f"Samples read:  {total}")
    print(f"GPX points:    {len(points)}")
    print(f"Skipped:       {total - len(points)}")
    print(f"First UTC:     {points[0].timestamp}")
    print(f"Last UTC:      {points[-1].timestamp}")
    metrics = []
    if do_exposure:
        metrics.extend((("ISO", "iso"), ("Shutter", "shutter_seconds"),
                        ("F-number", "f_number"),
                        ("Color temperature", "color_temperature_k")))
    if do_flight_telemetry:
        metrics.extend((("Roll", "roll_deg"), ("Pitch", "pitch_deg"),
                        ("Yaw", "yaw_deg"), ("Gimbal yaw", "gimbal_yaw_deg"),
                        ("Altitude", "altitude_m")))
    for label, attr in metrics:
        available = sum(getattr(point, attr) is not None for point in points)
        print(f"{label + ' points:':<15}{available} / {len(points)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())