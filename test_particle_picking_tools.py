#!/usr/bin/env python3
"""Test script for particle picking tools."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cryoagent.config.config_loader import ConfigLoader, RELIONSettings
from cryoagent.tools.relion_tools import RELIONTools

def check_job_completion_files(output_dir: str) -> str:
        """
        Check for RELION job completion files in the output directory.
        
        Args:
            output_dir: Path to the job output directory
            
        Returns:
            Status string: 'completed', 'failed', or 'running'
        """
        if not os.path.exists(output_dir):
            return "unknown"
        
        success_file = os.path.join(output_dir, "RELION_JOB_EXIT_SUCCESS")
        failure_file = os.path.join(output_dir, "RELION_JOB_EXIT_FAILURE")
        
        if os.path.exists(success_file):
            return "completed"
        elif os.path.exists(failure_file):
            return "failed"
        else:
            return "running"

RELION_CONFIG_PATH = "configs/relion/particle_picking_config.json"
CRYOSPARC_CONFIG_PATH = "configs/cryosparc/particle_picking_config.json"
PREPROCESSING_RESULTS_PATH = "outputs/preprocessing_results_relion_20251029_174012.json"


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON data from disk with basic error handling."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_relion_step_parameters(config: Dict[str, Any], step_name: str) -> Dict[str, Any]:
    """Fetch parameter dictionary for a specific RELION workflow step."""
    workflow = config.get("workflow", {})
    if step_name in workflow:
        return workflow[step_name].copy()
    raise KeyError(f"Step '{step_name}' not found in RELION particle picking config")


def merge_parameters(base: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge configuration parameters with overrides while keeping originals intact."""
    merged = base.copy()
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def test_particle_picking_tools():
    """Test all particle picking tools."""
    print("🧪 Testing Particle Picking Tools")
    print("=" * 50)
    
    # Load workflow configurations
    try:
        relion_config = load_json(RELION_CONFIG_PATH)
        cryosparc_config = load_json(CRYOSPARC_CONFIG_PATH)
        print(f"Loaded RELION config: {RELION_CONFIG_PATH}")
        print(f"Loaded CryoSPARC config: {CRYOSPARC_CONFIG_PATH}")
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Required configuration missing: {err}")

    relion_step_params = {
        "blob_picker": get_relion_step_parameters(relion_config, "blob_picker"),
        "particle_extraction": get_relion_step_parameters(relion_config, "particle_extraction"),
        "classification_2d": get_relion_step_parameters(relion_config, "classification_2d"),
        "auto_2d_selection": get_relion_step_parameters(relion_config, "auto_2d_selection"),
        "template_picker": get_relion_step_parameters(relion_config, "template_picker"),
    }

    # Load preprocessing results for realistic file paths
    preprocessing_results = {}
    if os.path.exists(PREPROCESSING_RESULTS_PATH):
        preprocessing_results = load_json(PREPROCESSING_RESULTS_PATH)
        print(f"Loaded preprocessing outputs: {PREPROCESSING_RESULTS_PATH}")
    else:
        print(f"⚠️ Preprocessing results file not found: {PREPROCESSING_RESULTS_PATH}")

    stage_outputs = preprocessing_results.get("stage_outputs", {})

    # Load configuration
    config_loader = ConfigLoader(
        config_path="configs/master_config.json",
        master_config_path="configs/master_config.json"
    )
    settings = config_loader.get_relion_settings()
    
    # Initialize RELION tools
    relion_tools = RELIONTools(settings, config_loader)
    
    # Test data paths derived from preprocessing results (fallback to defaults if missing)
    test_micrographs_star = stage_outputs.get(
        "selected_micrographs_star",
        "Select/job001/micrographs.star",
    )
    test_particles_star = stage_outputs.get(
        "particles_star_file",
        "Extract/job001/particles.star",
    )
    test_optimiser_star = stage_outputs.get(
        "class2d_optimiser_star",
        "Class2D/job001/run_optimiser.star",
    )

    print("Input files used for testing:")
    print(f"  Micrographs STAR: {test_micrographs_star}")
    print(f"  Particles STAR:   {test_particles_star}")
    print(f"  Optimiser STAR:   {test_optimiser_star}")

    blob_result: Optional[Dict[str, Any]] = None
    extraction_result: Optional[Dict[str, Any]] = None
    classification_result: Optional[Dict[str, Any]] = None
    auto_select_result: Optional[Dict[str, Any]] = None
    template_result: Optional[Dict[str, Any]] = None
    extraction_result2: Optional[Dict[str, Any]] = None
    classification_result2: Optional[Dict[str, Any]] = None
    
    print("\n1. Testing Blob Picker Tool")
    print("-" * 30)
    try:
        # Merge config parameters with test overrides
        blob_params = merge_parameters(
            relion_step_params["blob_picker"],
            {
                "wait_for_completion": False,
            },
        )
        blob_result = relion_tools.blob_picker(
            input_star=test_micrographs_star,
            output_dir="AutoPick",
            **blob_params,
        )
        print(f"✅ Blob picker test passed: {blob_result['job_type']}")
        print(f"   Output directory: {blob_result['output_dir']}")
    except Exception as e:
        print(f"❌ Blob picker test failed: {e}")
    
    print("\n2. Testing Particle Extraction Tool")
    print("-" * 30)
    try:
        # Verify key extraction parameters are loaded from config
        extraction_config = relion_step_params["particle_extraction"]
        print(f"   Loaded parameters:")
        print(f"     extract_size: {extraction_config.get('extract_size', 'N/A')}")
        print(f"     float16: {extraction_config.get('float16', 'N/A')}")
        print(f"     scale: {extraction_config.get('scale', 'N/A')}")
        print(f"     bg_radius: {extraction_config.get('bg_radius', 'N/A')}")
        print(f"     invert_contrast: {extraction_config.get('invert_contrast', 'N/A')}")
        print(f"     norm: {extraction_config.get('norm', 'N/A')}")
        
        particle_params = merge_parameters(
            relion_step_params["particle_extraction"],
            {
                "wait_for_completion": False,
                "coord_list": (blob_result or {}).get("output_dir", "AutoPick/job001"),
            },
        )
        extraction_result = relion_tools.particle_extraction(
            input_star=test_micrographs_star,
            output_dir="Extract",
            **particle_params,
        )
        print(f"✅ Particle extraction test passed: {extraction_result['job_type']}")
        print(f"   Output directory: {extraction_result['output_dir']}")
    except Exception as e:
        print(f"❌ Particle extraction test failed: {e}")
        print(f"   Input STAR attempted: {test_micrographs_star}")
    
    print("\n3. Testing 2D Classification Tool")
    print("-" * 30)
    try:
        classification_input = (
            os.path.join(extraction_result["output_dir"], "particles.star")
            if extraction_result
            else test_particles_star
        )
        class2d_params = merge_parameters(
            relion_step_params["classification_2d"],
            {
                "wait_for_completion": True
            },
        )
        classification_result = relion_tools.classification_2d(
            input_star=classification_input,
            output_dir="Class2D",
            **class2d_params,
        )
        print(f"✅ 2D classification test passed: {classification_result['job_type']}")
        print(f"   Output directory: {classification_result['output_dir']}")
    except Exception as e:
        print(f"❌ 2D classification test failed: {e}")
    
    print("\n4. Testing Auto 2D Selection Tool")
    print("-" * 30)
    try:
        optimiser_input = classification_result["optimiser_star"]
        auto_select_params = merge_parameters(
            relion_step_params["auto_2d_selection"],
            {
                "wait_for_completion": False,
            },
        )
        auto_select_result = relion_tools.auto_2d_selection(
            input_opt=optimiser_input,
            output_dir="Select",
            use_backend=True,
            **auto_select_params,
        )

        print(f"   Waiting for job completion: {auto_select_result['output_dir']}")
        while check_job_completion_files(auto_select_result["output_dir"]) != "completed":
            print(f"   Waiting for job completion: {auto_select_result['output_dir']}")
            time.sleep(5)

        print(f"✅ Auto 2D selection test passed: {auto_select_result['job_type']}")
        print(f"   Output directory: {auto_select_result['output_dir']}")
    except Exception as e:
        print(f"❌ Auto 2D selection test failed: {e}")
    
    print("\n5. Testing Template Picker Tool (Second Round)")
    print("-" * 30)
    try:
        ref_star_path = os.path.join(auto_select_result["output_dir"], "class_averages.star") if auto_select_result else "Select/job001/class_averages.star"
        template_params = merge_parameters(
            relion_step_params["template_picker"],
            {
                "wait_for_completion": False,
                "ref_star": ref_star_path,
            },
        )
        template_result = relion_tools.template_picker(
            input_star=test_micrographs_star,
            output_dir="AutoPick",
            use_backend=False,
            **template_params,
        )
        print(f"✅ Template picker test passed: {template_result['job_type']}")
        print(f"   Output directory: {template_result['output_dir']}")
    except Exception as e:
        print(f"❌ Template picker test failed: {e}")
    
    print("\n6. Testing Particle Extraction Tool (Second Round)")
    print("-" * 30)
    try:
        extraction_config = relion_step_params["particle_extraction"]
        particle_params = merge_parameters(
            extraction_config,
            {
                "wait_for_completion": False,
                "coord_list": (template_result or {}).get("output_dir"),
            },
        )
        extraction_result2 = relion_tools.particle_extraction(
            input_star=test_micrographs_star,
            output_dir="Extract",
            **particle_params,
        )
        print(f"✅ Particle extraction (2nd round) test passed: {extraction_result2['job_type']}")
        print(f"   Output directory: {extraction_result2['output_dir']}")
    except Exception as e:
        print(f"❌ Particle extraction (2nd round) test failed: {e}")
    
    print("\n7. Testing 2D Classification Tool (Second Round)")
    print("-" * 30)
    try:
        classification_input2 = (
            os.path.join(extraction_result2["output_dir"], "particles.star")
            if extraction_result2
            else test_particles_star
        )
        
        class2d_params = merge_parameters(
            relion_step_params["classification_2d"],
            {
                "wait_for_completion": False,
            },
        )
        classification_result2 = relion_tools.classification_2d(
            input_star=classification_input2,
            output_dir="Class2D",
            use_backend=False,
            **class2d_params,
        )
        print(f"✅ 2D classification (2nd round) test passed: {classification_result2['job_type']}")
        print(f"   Output directory: {classification_result2['output_dir']}")
    except Exception as e:
        print(f"❌ 2D classification (2nd round) test failed: {e}")
    
    print("\n8. Testing Auto 2D Selection Tool (Second Round)")
    print("-" * 30)
    try:
        optimiser_input2 = (
            classification_result2["optimiser_star"]
            if classification_result2
            else test_optimiser_star
        )
        
        auto_select_params = merge_parameters(
            relion_step_params["auto_2d_selection"],
            {
                "wait_for_completion": False,
            },
        )
        auto_select_result2 = relion_tools.auto_2d_selection(
            input_opt=optimiser_input2,
            output_dir="Select",
            use_backend=True,
            **auto_select_params,
        )

        print(f"   Waiting for job completion: {auto_select_result2['output_dir']}")
        while check_job_completion_files(auto_select_result2["output_dir"]) != "completed":
            print(f"   Waiting for job completion: {auto_select_result2['output_dir']}")
            time.sleep(5)

        print(f"✅ Auto 2D selection (2nd round) test passed: {auto_select_result2['job_type']}")
        print(f"   Output directory: {auto_select_result2['output_dir']}")
    except Exception as e:
        print(f"❌ Auto 2D selection (2nd round) test failed: {e}")
    
    print("\n9. Testing Configuration Loading")
    print("-" * 30)
    try:
        relion_stage_info = relion_config.get("stage_info", {})
        relion_stage_name = relion_stage_info.get("name", "Unknown")
        react_workflow_steps = relion_config.get("react_workflow", {}).get("steps", [])
        cryosparc_stage = cryosparc_config.get("stage_info", {}).get("name")
        print("✅ Configuration loaded successfully")
        print(f"   RELION stage: {relion_stage_name}")
        print(f"   RELION workflow steps: {len(react_workflow_steps)}")
        print(f"   CryoSPARC stage: {cryosparc_stage}")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
    
    print("\n10. Testing Tool Validation")
    print("-" * 30)
    try:
        # Test input validation
        validation_result = relion_tools.validate_inputs("star_file", test_micrographs_star)
        print(f"✅ Input validation test: {validation_result}")
        
        validation_result = relion_tools.validate_inputs("movies", "*.mrc")
        print(f"✅ Movies validation test: {validation_result}")
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
    
    print("\n11. Preprocessing Outputs Summary")
    print("-" * 30)
    if stage_outputs:
        for key, value in stage_outputs.items():
            print(f"   {key}: {value}")
    else:
        print("   No preprocessing outputs available to display.")

    print("\n" + "=" * 50)
    print("🎉 Particle picking tools testing completed!")
    print("Note: These are command validation tests. Actual execution requires valid input files.")


if __name__ == "__main__":
    test_particle_picking_tools()
