#!/usr/bin/env python3
"""
Flip (mirror) cryo-EM density maps in MRC format.

Examples:
  python flip_mrc.py in.mrc out_xflip.mrc --axis x
  python flip_mrc.py in.mrc out_yflip.mrc --axis y
  python flip_mrc.py in.mrc out_zflip.mrc --axis z
  python flip_mrc.py in.mrc out_xyflip.mrc --axis x y
"""

import argparse
import numpy as np
import mrcfile


AXIS_MAP = {
    "x": 2,  # numpy array order from mrcfile is typically (z, y, x)
    "y": 1,
    "z": 0,
}


def flip_volume(data: np.ndarray, axes):
    out = data
    for a in axes:
        out = np.flip(out, axis=AXIS_MAP[a])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_mrc", help="Input MRC map")
    p.add_argument("output_mrc", help="Output flipped MRC map")
    p.add_argument("--axis", nargs="+", choices=["x", "y", "z"], default=["x"],
                   help="Axis/axes to flip (space-separated), e.g. --axis x or --axis x y")
    args = p.parse_args()

    with mrcfile.open(args.input_mrc, permissive=True) as m:
        data = np.array(m.data, copy=True)  # (z, y, x)
        voxel_size = m.voxel_size
        header = m.header.copy()
        extended_header = m.extended_header

        flipped = flip_volume(data, args.axis)

        # Write output
        with mrcfile.new(args.output_mrc, overwrite=True) as out:
            out.set_data(flipped.astype(np.float32, copy=False))
            out.voxel_size = voxel_size

            # Preserve header fields where reasonable
            out.header.nxstart = header.nxstart
            out.header.nystart = header.nystart
            out.header.nzstart = header.nzstart
            out.header.origin = header.origin

            # Preserve extended header if present
            if extended_header is not None and len(extended_header) > 0:
                out.set_extended_header(extended_header)

            out.update_header_from_data()
            out.update_header_stats()

    print(f"Wrote flipped map: {args.output_mrc} (flipped along {args.axis})")


if __name__ == "__main__":
    main()
