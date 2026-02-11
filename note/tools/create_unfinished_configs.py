#!/usr/bin/env python3
"""
Script to read full_test.csv, filter rows, and create config files for unfinished datasets.
"""

import csv
import json
import os
import re
import sys
from pathlib import Path


def parse_pixel_size(pixel_str):
    """Extract numeric value from pixel spacing string (e.g., '1.2156 Å' -> 1.2156)"""
    if not pixel_str or pixel_str.strip() == '':
        return None
    # Remove Å symbol and extract number
    match = re.search(r'(\d+\.?\d*)', pixel_str.replace('Å', '').strip())
    if match:
        return float(match.group(1))
    return None


def parse_cs_mm(cs_str):
    """Extract numeric value from spherical aberration string"""
    if not cs_str or cs_str.strip() == '' or 'No CS value found' in cs_str:
        return None
    match = re.search(r'(\d+\.?\d*)', cs_str.strip())
    if match:
        return float(match.group(1))
    return None


def parse_dose(dose_str):
    """Extract numeric value from electron dose string"""
    if not dose_str or dose_str.strip() == '':
        return None
    match = re.search(r'(\d+\.?\d*)', dose_str.strip())
    if match:
        return float(match.group(1))
    return None


def parse_voltage(voltage_str):
    """Extract numeric value from voltage string"""
    if not voltage_str or voltage_str.strip() == '':
        return None
    match = re.search(r'(\d+)', str(voltage_str).strip())
    if match:
        return int(match.group(1))
    return None


def create_microscope_config(empair_id, pixel_size, voltage, cs_mm, dose, symmetry, data_type, gain_type, template_path):
    """Create microscope_config.json from template with updated parameters.
    When gain_type is not N/A: movies_path = .../micrographs/*.{data_type}, micrographs_path = null, gain_ref_path = .../gain/*.{gain_type}.
    When gain_type is N/A: movies_path = null, micrographs_path = .../micrographs/*.{data_type}, gain_ref_path = null.
    """
    with open(template_path, 'r') as f:
        config = json.load(f)
    
    # Update description
    config['microscope_info']['description'] = f"Basic microscope parameters for cryoEM data collection - Dataset {empair_id}"
    
    # Update parameters
    if pixel_size is not None:
        config['microscope_parameters']['pixel_size'] = pixel_size
    if voltage is not None:
        config['microscope_parameters']['voltage'] = voltage
    if cs_mm is not None:
        config['microscope_parameters']['cs_mm'] = cs_mm
    if dose is not None:
        config['microscope_parameters']['dose'] = dose
    if symmetry and symmetry.strip():
        config['microscope_parameters']['symmetry'] = symmetry.strip()
    
    # Paths under cryoPPP base
    base_path = f"/home/daoyi/storage_server/cryoPPP/cryoppp/{empair_id}"
    data_type = (data_type or "mrc").strip().lower()
    gain_type_raw = (gain_type or "").strip()
    gain_is_na = gain_type_raw.upper() == "N/A" or gain_type_raw == ""
    
    if gain_is_na:
        config['microscope_parameters']['movies_path'] = None
        config['microscope_parameters']['micrographs_path'] = f"{base_path}/micrographs/*.{data_type}"
        config['microscope_parameters']['gain_ref_path'] = None
    else:
        gain_ext = gain_type_raw.lower()
        config['microscope_parameters']['movies_path'] = f"{base_path}/micrographs/*.{data_type}"
        config['microscope_parameters']['micrographs_path'] = None
        config['microscope_parameters']['gain_ref_path'] = f"{base_path}/gain/*.{gain_ext}"
    
    return config


def main():
    # Paths
    csv_path = Path("note/full_test.csv")
    template_microscope = Path("datasets/finished_datasets/10002/configs/microscope_config.json")
    template_session = Path("configs/session.json")
    output_base = Path("datasets/unfinished_datasets")
    
    # Read CSV (using utf-8-sig to handle BOM)
    rows_to_process = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hetero = row.get('Hetero', '').strip()
            status = row.get('Status', '').strip()
            
            # Filter: Hetero is not Y and Status is not D or S
            if hetero != 'Y' and status not in ['D', 'S']:
                rows_to_process.append(row)
    
    print(f"Found {len(rows_to_process)} rows to process")
    
    # Process each row
    processed_count = 0
    for row in rows_to_process:
        try:
            empair_id = row.get('EMPAIR ID', '').strip()
            if not empair_id:
                continue
            
            print(f"\nProcessing {empair_id}...")
            
            # Extract parameters
            pixel_size = parse_pixel_size(row.get('Pixel Spacing (A)', ''))
            voltage = parse_voltage(row.get('Acceleration Voltage (kV)', ''))
            cs_mm = parse_cs_mm(row.get('Spherical Aberration (mm)', ''))
            dose = parse_dose(row.get('Electron Dose (e/A^2)', ''))
            symmetry = row.get('Point Group', '').strip()
            data_type = row.get('Data type', '').strip() or row.get('Data Type', '').strip()
            gain_type = row.get('gain type', '').strip() or row.get('Gain type', '').strip()
            
            # Create directory structure
            config_dir = output_base / empair_id / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Create microscope_config.json
            microscope_config = create_microscope_config(
                empair_id, pixel_size, voltage, cs_mm, dose, symmetry, data_type, gain_type, template_microscope
            )
            microscope_config_path = config_dir / "microscope_config.json"
            with open(microscope_config_path, 'w') as f:
                json.dump(microscope_config, f, indent=2)
            print(f"  Created: {microscope_config_path}")
            
            # Create session.json (heterogeneity and heterogeneity_depth false; polish true when gain type is not N/A)
            session_config_path = config_dir / "session.json"
            with open(template_session, 'r') as f:
                session_config = json.load(f)
            gain_is_na = (gain_type or "").strip().upper() == "N/A" or (gain_type or "").strip() == ""
            for stage in session_config.get("master_workflow", {}).get("stages", []):
                name = stage.get("name")
                if name == "heterogeneity" or name == "heterogeneity_depth":
                    stage["enabled"] = False
                elif name == "polish" and not gain_is_na:
                    stage["enabled"] = True
            with open(session_config_path, 'w') as f:
                json.dump(session_config, f, indent=2)
            print(f"  Created: {session_config_path}")
            processed_count += 1
        except Exception as e:
            print(f"  ERROR processing {empair_id}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    
    print(f"\nDone! Processed {processed_count} datasets.")


if __name__ == "__main__":
    main()

