#!/usr/bin/env python3
"""
Read note/full_test.csv and for each row where Status is not D or S:
1. Call EMPIAR API to get EMDB ID from cross_references and fill EMDB ID column.
2. Call EMDB API to get point_group (fill Point Group) and compute diameter
   from space.x and pixel_space.x: (d-128)*pixel_size, round up to 10, min 100.
"""

import csv
import math
import os
import sys
from pathlib import Path

import requests

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "note" / "full_test.csv"
EMPIAR_ENTRY_URL = "https://www.ebi.ac.uk/empiar/api/entry/{}"
EMDB_ENTRY_URL = "https://www.ebi.ac.uk/emdb/api/entry/{}"
EMDB_MAP_URL = "https://www.ebi.ac.uk/emdb/api/entry/map/{}"
REQUEST_TIMEOUT = 30


def should_process_row(status):
    """Process row only when Status is not D and not S."""
    if status is None:
        return True
    s = str(status).strip().upper()
    return s != "D" and s != "S"


def normalize_empiar_id(raw):
    """Return EMPIAR ID as string for API (digits only if numeric)."""
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw).strip()


def get_emdb_from_empiar(empiar_id):
    """
    Call EMPIAR entry API and return EMDB accession from cross_references.
    Returns None on failure or if not found.
    API may return { "EMPIAR-10389": { ... } } or the entry object directly.
    cross_references can be a list of strings (e.g. ["EMD-10835"]) or list of objects.
    """
    url = EMPIAR_ENTRY_URL.format(empiar_id)
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  EMPIAR API error for {empiar_id}: {e}", file=sys.stderr)
        return None

    # Unwrap if response is keyed by entry id (e.g. {"EMPIAR-10389": { ... }})
    if isinstance(data, dict) and len(data) == 1:
        key = next(iter(data))
        if key and str(key).upper().startswith("EMPIAR-"):
            data = data[key]
    if not isinstance(data, dict):
        return None

    # Try top-level EMDB fields first
    for key in ("emdb_id", "emdb_accession", "related_emdb"):
        val = data.get(key)
        if val and str(val).strip():
            return str(val).strip()

    refs = data.get("cross_references") or data.get("crossReferences") or []
    if not isinstance(refs, list):
        refs = [refs] if refs else []
    for ref in refs:
        # EMPIAR API can return cross_references as list of strings: ["EMD-10835"]
        if isinstance(ref, str):
            s = ref.strip()
            if s.upper().startswith("EMD-") or (s.isdigit() and len(s) <= 6):
                return s
            continue
        if not isinstance(ref, dict):
            continue
        db = (ref.get("database") or ref.get("db") or ref.get("type") or "").upper()
        acc = ref.get("accession") or ref.get("acc") or ref.get("id") or ref.get("accession_id")
        if acc and ("EMDB" in db or "EMD" in str(acc)):
            acc_str = str(acc).strip()
            if acc_str:
                return acc_str
    return None


def normalize_emdb_id_for_url(emdb_id):
    """EMDB API URL: strip EMD- prefix only; do not strip leading zeros (e.g. 0407 stays 0407)."""
    if not emdb_id:
        return None
    s = str(emdb_id).strip()
    if s.upper().startswith("EMD-"):
        s = s[4:].strip()
    return s or None


def _extract_from_json(obj, *key_paths):
    """Try each key path (e.g. ('map', 'space', 'x')) and return first non-None."""
    for path in key_paths:
        cur = obj
        for k in path:
            if not isinstance(cur, dict):
                break
            cur = cur.get(k)
        if cur is not None and cur != "":
            return cur
    return None


def _emdb_value(val):
    """EMDB API often wraps values in { 'valueOf_': <value>, 'units': '...' }. Unwrap."""
    if isinstance(val, dict):
        return val.get("valueOf_") or val.get("value") or val.get("id")
    return val


def get_emdb_entry(emdb_id):
    """
    Call EMDB entry API. Extract point_group, dimension, pixel_size from the
    structure_determination_list / map / interpretation layout (see EMDB API schema).
    """
    url_id = normalize_emdb_id_for_url(emdb_id)
    if not url_id:
        return None
    url = EMDB_ENTRY_URL.format(url_id)
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  EMDB API error for {emdb_id}: {e}", file=sys.stderr)
        return None

    # point_group: structure_determination_list.structure_determination[].image_processing[].final_reconstruction.applied_symmetry.point_group
    point_group = ""
    sdl = data.get("structure_determination_list") or {}
    sd_list = sdl.get("structure_determination")
    if isinstance(sd_list, list):
        for sd in sd_list:
            ip_list = (sd.get("image_processing") or []) if isinstance(sd, dict) else []
            if not isinstance(ip_list, list):
                ip_list = [ip_list] if ip_list else []
            for ip in ip_list:
                if not isinstance(ip, dict):
                    continue
                fr = ip.get("final_reconstruction") or {}
                asym = fr.get("applied_symmetry") if isinstance(fr, dict) else None
                if isinstance(asym, dict):
                    pg = asym.get("point_group")
                    if pg is not None and pg != "":
                        point_group = _emdb_value(pg) or pg
                        if isinstance(point_group, str):
                            point_group = point_group.strip()
                        else:
                            point_group = str(point_group).strip()
                        break
            if point_group:
                break
    if not point_group:
        pg = _extract_from_json(data, ("point_group",), ("map", "point_group"))
        if pg is not None:
            point_group = str(_emdb_value(pg) or pg).strip()

    # dimension d: map.dimensions.col / .row / .sec or map.spacing.x (x dimension)
    dimension = _extract_from_json(
        data,
        ("map", "dimensions", "col"),
        ("map", "dimensions", "x"),
        ("map", "spacing", "x"),
        ("map", "dimensions", "row"),
        ("map", "dimensions", "sec"),
    )
    if dimension is not None and isinstance(dimension, dict):
        dimension = _emdb_value(dimension)
    # pixel_size: map.pixel_spacing.x.valueOf_
    pixel_size = _extract_from_json(
        data,
        ("map", "pixel_spacing", "x"),
        ("map", "pixel_space", "x"),
    )
    if pixel_size is not None and isinstance(pixel_size, dict):
        pixel_size = _emdb_value(pixel_size)

    # Fallback: interpretation.half_map_list.half_map[0].dimensions / pixel_spacing
    if dimension is None or pixel_size is None:
        interp = data.get("interpretation") or {}
        hm_list = interp.get("half_map_list", {}).get("half_map") if isinstance(interp, dict) else []
        if isinstance(hm_list, list) and len(hm_list) > 0:
            hm = hm_list[0]
            if isinstance(hm, dict):
                if dimension is None:
                    dims = hm.get("dimensions") or hm.get("spacing")
                    if isinstance(dims, dict):
                        dimension = dims.get("col") or dims.get("x")
                    if dimension is not None and isinstance(dimension, dict):
                        dimension = _emdb_value(dimension)
                if pixel_size is None:
                    ps = hm.get("pixel_spacing") or hm.get("pixel_space")
                    if isinstance(ps, dict):
                        pixel_size = ps.get("x")
                    if pixel_size is not None and isinstance(pixel_size, dict):
                        pixel_size = _emdb_value(pixel_size)

    # If still missing, try map endpoint
    if (dimension is None or pixel_size is None) and url_id:
        try:
            r2 = requests.get(EMDB_MAP_URL.format(url_id), timeout=REQUEST_TIMEOUT)
            if r2.ok:
                map_data = r2.json()
                if dimension is None:
                    dimension = _extract_from_json(
                        map_data,
                        ("map", "dimensions", "col"),
                        ("map", "dimensions", "x"),
                        ("map", "spacing", "x"),
                        ("dimensions", "col"),
                        ("dimensions", "x"),
                    )
                    if dimension is not None and isinstance(dimension, dict):
                        dimension = _emdb_value(dimension)
                if pixel_size is None:
                    pixel_size = _extract_from_json(
                        map_data,
                        ("map", "pixel_spacing", "x"),
                        ("map", "pixel_space", "x"),
                        ("pixel_spacing", "x"),
                    )
                    if pixel_size is not None and isinstance(pixel_size, dict):
                        pixel_size = _emdb_value(pixel_size)
        except Exception:
            pass

    try:
        dimension = int(float(dimension)) if dimension is not None else None
    except (TypeError, ValueError):
        dimension = None
    try:
        pixel_size = float(pixel_size) if pixel_size is not None else None
    except (TypeError, ValueError):
        pixel_size = None

    return {
        "point_group": str(point_group).strip() if point_group else "",
        "dimension": dimension,
        "pixel_size": pixel_size,
    }


def compute_diameter(dimension, pixel_size):
    """
    Diameter = (d - 128) * pixel_size, round up to nearest 10, minimum 100.
    """
    if dimension is None or pixel_size is None:
        return None
    try:
        d = int(dimension)
        ps = float(pixel_size)
    except (TypeError, ValueError):
        return None
    raw = (d - 128) * ps
    rounded = math.ceil(raw / 10.0) * 10
    return max(100, int(rounded))


def main():
    import argparse
    p = argparse.ArgumentParser(description="Fill EMDB ID, Point Group, Diameter from EMPIAR/EMDB APIs")
    p.add_argument("--dry-run", action="store_true", help="Do not write CSV back")
    args = p.parse_args()

    if not CSV_PATH.is_file():
        print(f"CSV not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        print("No rows in CSV.")
        return

    def write_csv():
        if args.dry_run:
            return
        with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    for i, row in enumerate(rows):
        status = row.get("Status", "")
        if not should_process_row(status):
            continue

        empair_id_raw = row.get("EMPAIR ID", "")
        empiar_id = normalize_empiar_id(empair_id_raw)
        if not empiar_id:
            print(f"Row {i + 2}: skip (no EMPAIR ID)")
            continue

        emdb_id = (row.get("EMDB ID") or "").strip()
        if not emdb_id:
            print(f"Row {i + 2}: EMPIAR {empiar_id} -> fetching EMDB ID...")
            emdb_id = get_emdb_from_empiar(empiar_id)
            if emdb_id:
                row["EMDB ID"] = emdb_id
                print(f"  EMDB ID: {emdb_id}")
                write_csv()
            else:
                print(f"  No EMDB cross-reference in EMPIAR (entry may not be deposited to EMDB yet)")
                continue
        else:
            print(f"Row {i + 2}: using existing EMDB ID {emdb_id}")

        info = get_emdb_entry(emdb_id)
        if not info:
            print(f"  Could not get EMDB entry")
            continue

        if info.get("point_group") and (not (row.get("Point Group") or "").strip()):
            row["Point Group"] = info["point_group"]
            print(f"  Point Group: {info['point_group']}")

        diameter = compute_diameter(info.get("dimension"), info.get("pixel_size"))
        if diameter is not None:
            if not (row.get("Diameter") or "").strip():
                row["Diameter"] = str(diameter)
                print(f"  Diameter: {diameter} (dim={info.get('dimension')}, px={info.get('pixel_size')})")
            else:
                print(f"  Diameter: {diameter} (already set)")
        else:
            print(f"  Diameter: could not compute (dim={info.get('dimension')}, px={info.get('pixel_size')})")

        write_csv()

    if args.dry_run:
        print("(Dry run: not writing CSV)")
    else:
        print("Done. CSV updated after each row.")


if __name__ == "__main__":
    main()
