#!/usr/bin/env python3
"""Calculate time difference between two ISO-format timestamps."""

import argparse
from datetime import datetime


def parse_ts(s: str) -> datetime:
    """Parse ISO timestamp (e.g. 2026-01-22T05:44:06.820561)."""
    # Strip whitespace and handle optional 'Z' or timezone
    s = s.strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(s.replace("Z", ""), fmt.replace(".%fZ", ".%f").replace("Z", ""))
        except ValueError:
            continue
    raise ValueError(f"Could not parse timestamp: {s!r}")


def format_duration(seconds: float) -> str:
    """Format duration in seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.3f} seconds"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.3f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {secs:.3f}s"


# Default timestamps (start and end)
DEFAULT_START = "2026-01-22T05:44:06.820561"
DEFAULT_END = "2026-01-22T06:39:08.464510"


def main():
    parser = argparse.ArgumentParser(
        description="Calculate time between two ISO timestamps (e.g. 2026-01-22T05:44:06.820561)"
    )
    parser.add_argument(
        "start",
        nargs="?",
        default=DEFAULT_START,
        help=f"Start time (ISO format). Default: {DEFAULT_START}",
    )
    parser.add_argument(
        "end",
        nargs="?",
        default=DEFAULT_END,
        help=f"End time (ISO format). Default: {DEFAULT_END}",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Only print duration in seconds",
    )
    args = parser.parse_args()

    start = parse_ts(args.start)
    end = parse_ts(args.end)
    delta = end - start
    total_seconds = delta.total_seconds()

    if total_seconds < 0:
        raise SystemExit("Error: end time is before start time.")

    total_minutes = total_seconds / 60

    if args.quiet:
        print(f"{total_minutes:.6f}")
    else:
        print(f"Start:  {args.start}")
        print(f"End:    {args.end}")
        print(f"Delta:  {total_minutes:.1f} minutes ({format_duration(total_seconds)})")


if __name__ == "__main__":
    main()
