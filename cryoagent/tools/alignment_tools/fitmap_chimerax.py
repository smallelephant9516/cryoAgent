#!/usr/bin/env chimerax
"""
Fit a source density map into a target density map using ChimeraX fitmap,
then resample onto the target grid and save the fitted source as MRC.

Usage (recommended):
  chimerax --nogui --script "fitmap_map_to_map.py <source_map.mrc> <target_map.mrc> <output_fitted_source.mrc> [resolution_A]"

Example:
  chimerax --nogui --script "fitmap_map_to_map.py source.mrc target.mrc fitted_source.mrc 8.0"

Notes:
- Saving a moved map directly will NOT preserve rotation/translation in the output file format.
  Therefore we resample onto the target grid before saving. :contentReference[oaicite:1]{index=1}
"""

import os
import sys

from chimerax.core.commands import run


def _parse_args(argv):
    # ChimeraX passes script arguments after the script name in the --script string.
    # We keep parsing simple and robust.
    if len(argv) < 4:
        raise SystemExit(
            "ERROR: Not enough arguments.\n"
            "Usage:\n"
            "  chimerax --nogui --script \"fitmap_map_to_map.py <source_map> <target_map> <output_map> [resolution_A]\""
        )
    source_map = argv[1]
    target_map = argv[2]
    out_map = argv[3]
    resolution = float(argv[4]) if len(argv) >= 5 else None
    return source_map, target_map, out_map, resolution


def _abs_path(p):
    return os.path.abspath(os.path.expanduser(p))


source_path, target_path, out_path, resolution = _parse_args(sys.argv)

source_path = _abs_path(source_path)
target_path = _abs_path(target_path)

out_path = _abs_path(out_path)
print('start fitting map')

if not os.path.isfile(target_path):
    raise SystemExit(f"ERROR: target map not found: {target_path}")
if not os.path.isfile(source_path):
    raise SystemExit(f"ERROR: source map not found: {source_path}")

# Open maps

run(session, f'open "{source_path}"')
source_model = session.models.list()[-1]
source_id = source_model.id_string

run(session, f'open "{target_path}"')
target_model = session.models.list()[-1]  # last opened
target_id = target_model.id_string

# Fit source map into target map
# ChimeraX syntax: fitmap <movable> in <target> [options]
# "resolution" helps when fitting an atomic model; for map-to-map it can still be used
# to set a low-pass for correlation in some workflows. Keep optional.

source_id = 1
target_id = 2

if resolution is None:
    run(session, f"fitmap #{target_id} in #{source_id}")
else:
    run(session, f"fitmap #{target_id} in #{source_id} resolution {resolution}")
# IMPORTANT: saving a moved map won't preserve the transform; resample onto target grid first :contentReference[oaicite:2]{index=2}
# This creates a NEW map model (typically next model number).
run(session, f"volume resample #{target_id} onGrid #{source_id}")
resampled_model = session.models.list()[-1]
resampled_id = resampled_model.id_string
resampled_id = 3
# Save fitted/resampled source map
run(session, f'save "{out_path}" model #{resampled_id}')
# Optional: quit (useful for --nogui batch)
run(session, "quit")

# ChimeraX runs main(session) when opening a script


