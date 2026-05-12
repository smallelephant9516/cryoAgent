#!/usr/bin/env python3
"""
Batch Dataset Runner for CryoAgent

This script runs the CryoAgent workflow on multiple datasets continuously.
Each dataset should have its own folder in the datasets/unfinished_datasets/ directory with:
- configs/session.json
- configs/microscope_config.json
- configs/master_config.json (can be a symlink to the main one)

After successful completion, datasets are automatically moved from unfinished_datasets/ to finished_datasets/.

Usage:
    python run_batch_datasets.py [options]

Options:
    --datasets-dir DIR       Directory containing dataset folders (default: datasets/unfinished_datasets/)
    --workflow WORKFLOW      Workflow type: complete, preprocessing, custom (default: complete)
    --stages STAGES          Comma-separated list of stages for custom workflow
    --verbose                Enable verbose output
    --dry-run                Show what would be done without executing
    --datasets LIST          Comma-separated list of specific datasets to run (default: all)
    --continue-on-error      Continue processing next dataset if one fails
    --max-retries N          Maximum number of retries per dataset (default: 1)
"""

import sys
import argparse
import time
import logging
import json
import shutil
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import subprocess

# Add the project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def setup_logging(verbose: bool = False, log_file: str = "batch_runner.log"):
    """Setup logging for the batch runner."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger("BatchRunner")


def find_datasets(datasets_dir: Path, dataset_names: Optional[List[str]] = None) -> List[Path]:
    """
    Find all dataset folders in the datasets directory.
    
    Args:
        datasets_dir: Directory containing dataset folders
        dataset_names: Optional list of specific dataset names to process
        
    Returns:
        List of dataset folder paths
    """
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")
    
    datasets = []
    
    if dataset_names:
        # Process only specified datasets
        for name in dataset_names:
            dataset_path = datasets_dir / name
            if dataset_path.exists() and dataset_path.is_dir():
                datasets.append(dataset_path)
            else:
                logging.warning(f"Dataset folder not found: {dataset_path}")
    else:
        # Find all dataset folders
        for item in datasets_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if it has the required config structure
                if (item / "configs" / "session.json").exists():
                    datasets.append(item)
                else:
                    logging.warning(f"Skipping {item.name}: missing configs/session.json")
    
    return sorted(datasets)


def setup_dataset_configs(dataset_path: Path, project_root: Path, logger: logging.Logger) -> tuple[Path, bool]:
    """
    Set up configuration files for a dataset by creating a temporary config structure.
    
    The user only provides session.json and microscope_config.json. This function
    dynamically copies all other configs from the main configs directory.
    
    Args:
        dataset_path: Path to the dataset folder
        project_root: Path to the project root
        logger: Logger instance
        
    Returns:
        Tuple of (temp_config_path, success). temp_config_path is the path to the
        temporary config directory that should be used for the workflow.
    """
    try:
        dataset_configs = dataset_path / "configs"
        
        # Check required files exist
        required_files = ["session.json", "microscope_config.json"]
        for req_file in required_files:
            if not (dataset_configs / req_file).exists():
                logger.error(f"Missing required file: {dataset_configs / req_file}")
                return None, False
        
        # Create temporary config directory in dataset folder
        temp_config_dir = dataset_path / "configs_temp"
        
        # Clean up any existing temp directory
        if temp_config_dir.exists():
            shutil.rmtree(temp_config_dir)
        
        temp_config_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created temporary config directory: {temp_config_dir}")
        
        # Copy master_config.json from main configs
        main_master_config = project_root / "configs" / "master_config.json"
        if not main_master_config.exists():
            logger.error(f"Main master_config.json not found: {main_master_config}")
            return None, False
        
        shutil.copy2(main_master_config, temp_config_dir / "master_config.json")
        
        # Copy dataset-specific session.json to temp config
        # (The workflow looks for session.json in the same directory as master_config.json)
        shutil.copy2(dataset_configs / "session.json", temp_config_dir / "session.json")
        
        # Note: cryosparc and relion configs are shared and already exist at
        # project_root/configs/cryosparc/ and project_root/configs/relion/
        # They will be found by the workflow when it runs from project_root
        logger.debug(f"Cryosparc and relion configs will be loaded from {project_root / 'configs'}")
        
        # Copy dataset-specific microscope_config.json to main configs/ for workflow to find
        # (The workflow looks for it at project_root/configs/microscope_config.json)
        main_configs_dir = project_root / "configs"
        main_microscope_config = main_configs_dir / "microscope_config.json"
        dataset_microscope_config = dataset_configs / "microscope_config.json"
        
        # Backup existing microscope_config.json if it exists and is different
        backup_path = None
        if main_microscope_config.exists():
            try:
                with open(main_microscope_config, 'r') as f:
                    existing_content = json.load(f)
                with open(dataset_microscope_config, 'r') as f:
                    new_content = json.load(f)
                
                # Only backup if different
                if existing_content != new_content:
                    backup_path = main_configs_dir / "microscope_config.json.backup"
                    shutil.copy2(main_microscope_config, backup_path)
                    logger.debug(f"Backed up existing microscope_config.json to {backup_path}")
            except Exception as e:
                logger.warning(f"Could not compare microscope configs: {e}")
        
        # Copy dataset-specific microscope_config.json to main location
        shutil.copy2(dataset_microscope_config, main_microscope_config)
        logger.info(f"Set microscope_config.json for dataset {dataset_path.name}")
        
        # Store backup path in temp directory for cleanup
        if backup_path:
            (temp_config_dir / ".backup_path").write_text(str(backup_path))
        
        return temp_config_dir, True
        
    except Exception as e:
        logger.error(f"Failed to setup configs for {dataset_path.name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, False


def cleanup_dataset_configs(temp_config_dir: Path, project_root: Path, logger: logging.Logger) -> None:
    """
    Clean up temporary config directory and restore microscope_config.json backup if exists.
    
    Args:
        temp_config_dir: Path to temporary config directory
        project_root: Path to the project root
        logger: Logger instance
    """
    try:
        if not temp_config_dir.exists():
            return
        
        # Restore microscope_config.json backup if it exists
        backup_path_file = temp_config_dir / ".backup_path"
        if backup_path_file.exists():
            backup_path = Path(backup_path_file.read_text().strip())
            main_microscope_config = project_root / "configs" / "microscope_config.json"
            
            if backup_path.exists() and main_microscope_config.exists():
                shutil.copy2(backup_path, main_microscope_config)
                backup_path.unlink()
                logger.debug(f"Restored microscope_config.json from backup")
        
        # Remove temporary config directory
        shutil.rmtree(temp_config_dir)
        logger.debug(f"Cleaned up temporary config directory: {temp_config_dir}")
        
    except Exception as e:
        logger.warning(f"Failed to cleanup configs: {e}")



def check_workflow_summary_exists(dataset_output_dir: Path, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Check if a successful workflow summary report exists for the dataset.
    
    Args:
        dataset_output_dir: Directory containing dataset outputs
        logger: Logger instance
        
    Returns:
        Dictionary with summary report info if successful workflow found, None otherwise
    """
    if not dataset_output_dir.exists():
        return None
    
    # Look for workflow summary reports
    summary_files = list(dataset_output_dir.glob("workflow_summary_report_*.json"))
    
    if not summary_files:
        return None
    
    # Get the most recent summary report
    latest_summary = max(summary_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest_summary, 'r') as f:
            summary_data = json.load(f)
        
        # Check if it's a valid workflow summary
        if summary_data.get("report_type") != "cryoagent_workflow_summary":
            return None
        
        workflow_metadata = summary_data.get("workflow_metadata", {})
        executive_summary = summary_data.get("executive_summary", {})
        
        # Check if workflow completed successfully
        overall_status = executive_summary.get("overall_status", "")
        successful_stages = workflow_metadata.get("successful_stages", 0)
        total_stages = workflow_metadata.get("total_stages", 0)
        
        if overall_status == "success" and successful_stages == total_stages and total_stages > 0:
            return {
                "file_path": str(latest_summary),
                "timestamp": summary_data.get("timestamp", ""),
                "conversation_id": summary_data.get("conversation_id", ""),
                "successful_stages": successful_stages,
                "total_stages": total_stages,
                "overall_status": overall_status
            }
    except (json.JSONDecodeError, IOError, KeyError) as e:
        logger.debug(f"Failed to read workflow summary {latest_summary}: {e}")
    
    return None


def move_dataset_to_finished(
    dataset_path: Path,
    unfinished_dir: Path,
    finished_dir: Path,
    logger: logging.Logger
) -> bool:
    """
    Move a completed dataset folder from unfinished_datasets to finished_datasets.
    
    Args:
        dataset_path: Path to the dataset folder (currently in unfinished_datasets)
        unfinished_dir: Directory containing unfinished datasets
        finished_dir: Directory to move completed datasets to (finished_datasets)
        logger: Logger instance
        
    Returns:
        True if move was successful, False otherwise
    """
    try:
        # Check if dataset is actually in the unfinished directory
        try:
            dataset_path.relative_to(unfinished_dir)
        except ValueError:
            # Dataset is not in unfinished_dir, skip moving
            logger.debug(f"Dataset {dataset_path.name} is not in unfinished_datasets, skipping move")
            return True
        
        # Ensure finished directory exists
        finished_dir.mkdir(parents=True, exist_ok=True)
        
        # Destination path
        destination = finished_dir / dataset_path.name
        
        # Check if destination already exists
        if destination.exists():
            logger.warning(f"Destination {destination} already exists. Skipping move.")
            return False
        
        # Move the entire dataset folder
        shutil.move(str(dataset_path), str(destination))
        logger.info(f"✅ Moved completed dataset {dataset_path.name} from unfinished_datasets to finished_datasets/")
        logger.info(f"   Source: {dataset_path}")
        logger.info(f"   Destination: {destination}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to move dataset {dataset_path.name} to finished: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def check_all_stages_completed(dataset_output_dir: Path, workflow_type: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Check if all required stages for a workflow are completed by checking individual stage result files.
    This is a fallback when no workflow summary report exists.
    
    Args:
        dataset_output_dir: Directory containing dataset outputs
        workflow_type: Type of workflow (complete, preprocessing, etc.)
        logger: Logger instance
        
    Returns:
        Dictionary with completion info if all stages completed, None otherwise
    """
    if not dataset_output_dir.exists():
        return None
    
    # Map workflow types to required stage patterns
    # These match the patterns used in cryoagent_workflow.py
    stage_patterns_map = {
        "complete": [
            "preprocessing_results_*.json",
            "particle_picking_results_*.json",
            "2d_optimization_results_*.json",
            "reconstruction_results_*.json",
            "optimization_results_*.json"
        ],
        "preprocessing": [
            "preprocessing_results_*.json"
        ]
    }
    
    # Get required patterns for this workflow type
    required_patterns = stage_patterns_map.get(workflow_type, [])
    
    if not required_patterns:
        # Unknown workflow type, can't check
        return None
    
    completed_stages = []
    failed_stages = []
    stage_info = {}
    
    # Check each required stage
    for pattern in required_patterns:
        matching_files = list(dataset_output_dir.glob(pattern))
        
        if not matching_files:
            # Stage result file doesn't exist
            return None
        
        # Get the most recent file for this stage
        latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                stage_data = json.load(f)
            
            status = stage_data.get("status", "")
            stage_name = pattern.replace("_results_*.json", "").replace("_", " ")
            
            if status == "completed":
                completed_stages.append(stage_name)
                stage_info[stage_name] = {
                    "status": "completed",
                    "timestamp": stage_data.get("timestamp", ""),
                    "file": latest_file.name
                }
            else:
                # Stage exists but not completed (failed, etc.)
                failed_stages.append(stage_name)
                return None  # Can't consider workflow complete if any stage failed
                
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.debug(f"Failed to read stage result file {latest_file}: {e}")
            return None
    
    # All required stages are completed
    return {
        "method": "individual_stage_files",
        "completed_stages": completed_stages,
        "total_stages": len(required_patterns),
        "stage_info": stage_info
    }


def run_workflow_for_dataset(
    dataset_path: Path,
    project_root: Path,
    workflow_type: str = "complete",
    stages: Optional[str] = None,
    verbose: bool = False,
    dry_run: bool = False,
    logger: logging.Logger = None
) -> bool:
    """
    Run the CryoAgent workflow for a specific dataset.
    
    Args:
        dataset_path: Path to the dataset folder
        project_root: Path to the project root
        workflow_type: Type of workflow to run
        stages: Comma-separated list of stages (for custom workflow)
        verbose: Enable verbose output
        dry_run: Show what would be done without executing
        logger: Logger instance
        
    Returns:
        True if workflow completed successfully, False otherwise
    """
    if logger is None:
        logger = logging.getLogger("BatchRunner")
    
    dataset_name = dataset_path.name
    logger.info(f"{'='*80}")
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"{'='*80}")
    
    # Create dataset-specific output directory
    dataset_output_dir = dataset_path / "outputs"
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {dataset_output_dir}")
    
    # Check if workflow summary report exists and indicates successful completion
    existing_summary = check_workflow_summary_exists(dataset_output_dir, logger)
    if existing_summary:
        logger.info(f"✅ Found successful workflow summary for dataset {dataset_name}")
        logger.info(f"   📄 Summary file: {Path(existing_summary['file_path']).name}")
        logger.info(f"   📅 Completed: {existing_summary['timestamp']}")
        logger.info(f"   🎯 Status: {existing_summary['overall_status']}")
        logger.info(f"   ✅ Stages: {existing_summary['successful_stages']}/{existing_summary['total_stages']} completed")
        logger.info(f"   💬 Conversation ID: {existing_summary['conversation_id']}")
        logger.info(f"   ℹ️  Skipping workflow execution - dataset already completed successfully")
        return True
    
    # Fallback: Check individual stage result files if no summary report exists
    logger.debug(f"No workflow summary report found, checking individual stage result files...")
    stage_check_result = check_all_stages_completed(dataset_output_dir, workflow_type, logger)
    if stage_check_result:
        logger.info(f"✅ Found all required stages completed for dataset {dataset_name} (checked individual files)")
        logger.info(f"   📊 Method: {stage_check_result['method']}")
        logger.info(f"   ✅ Completed stages: {', '.join(stage_check_result['completed_stages'])}")
        logger.info(f"   📈 Total: {stage_check_result['total_stages']} stages completed")
        logger.info(f"   ℹ️  Skipping workflow execution - all stages already completed")
        return True
    
    # Setup configs - this creates a temporary config structure with all necessary files
    temp_config_dir, success = setup_dataset_configs(dataset_path, project_root, logger)
    if not success or temp_config_dir is None:
        logger.error(f"Failed to setup configs for {dataset_name}")
        return False
    
    # Path to temporary master_config.json (contains copied master + dataset session.json)
    temp_master_config = temp_config_dir / "master_config.json"
    
    if not temp_master_config.exists():
        logger.error(f"master_config.json not found in temp config for dataset {dataset_name}")
        cleanup_dataset_configs(temp_config_dir, project_root, logger)
        return False
    
    # Build command to run workflow
    workflow_script = project_root / "cryoagent_workflow.py"
    
    # Generate unique conversation ID that includes dataset name to ensure fresh start
    conversation_id_suffix = f"{int(time.time())}"
    unique_conversation_id = f"{dataset_name}_{workflow_type}_{conversation_id_suffix}"
    
    cmd = [
        sys.executable,
        str(workflow_script),
        "--config", str(temp_master_config),
        "--workflow", workflow_type,
        "--conversation-id", unique_conversation_id,
        "--outputs-dir", str(dataset_output_dir)
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    if dry_run:
        cmd.append("--dry-run")
    
    if stages and workflow_type == "custom":
        cmd.extend(["--stages", stages])
    
    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info(f"Conversation ID: {unique_conversation_id}")
    logger.info(f"Outputs directory: {dataset_output_dir}")
    
    # Change to project root directory to run workflow
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            check=False,
            capture_output=False,  # Let output stream to console
            text=True
        )
        
        elapsed_time = time.time() - start_time
        
        success = result.returncode == 0
        
        if success:
            logger.info(f"✅ Dataset {dataset_name} completed successfully in {elapsed_time:.1f} seconds")
        else:
            logger.error(f"❌ Dataset {dataset_name} failed with return code {result.returncode} after {elapsed_time:.1f} seconds")
        
        # Cleanup temporary configs
        cleanup_dataset_configs(temp_config_dir, project_root, logger)
        
        return success
            
    except KeyboardInterrupt:
        logger.warning(f"⚠️ Workflow for {dataset_name} interrupted by user")
        cleanup_dataset_configs(temp_config_dir, project_root, logger)
        raise
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ Unexpected error processing {dataset_name} after {elapsed_time:.1f} seconds: {e}")
        import traceback
        logger.error(traceback.format_exc())
        cleanup_dataset_configs(temp_config_dir, project_root, logger)
        return False


def main():
    """Main function to run batch processing of datasets."""
    parser = argparse.ArgumentParser(
        description="Batch Dataset Runner for CryoAgent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all datasets with complete workflow
  python run_batch_datasets.py

  # Run specific datasets
  python run_batch_datasets.py --datasets 10028,10204

  # Run with preprocessing only
  python run_batch_datasets.py --workflow preprocessing

  # Dry run to see what would be done
  python run_batch_datasets.py --dry-run

  # Verbose output
  python run_batch_datasets.py --verbose

  # Continue processing even if one dataset fails
  python run_batch_datasets.py --continue-on-error
        """
    )
    
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=project_root / "datasets" / "unfinished_datasets",
        help="Directory containing dataset folders (default: datasets/unfinished_datasets/)"
    )
    
    parser.add_argument(
        "--workflow",
        choices=["complete", "preprocessing", "custom", "test"],
        default="complete",
        help="Workflow type (default: complete)"
    )
    
    parser.add_argument(
        "--stages",
        help="Comma-separated list of stages for custom workflow"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    
    parser.add_argument(
        "--datasets",
        help="Comma-separated list of specific datasets to run (default: all)"
    )
    
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing next dataset if one fails"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum number of retries per dataset (default: 1)"
    )
    
    parser.add_argument(
        "--log-file",
        default="batch_runner.log",
        help="Log file path (default: batch_runner.log)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose, args.log_file)
    
    logger.info("Starting batch dataset runner")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Datasets directory: {args.datasets_dir}")
    
    # Parse dataset names if provided
    dataset_names = None
    if args.datasets:
        dataset_names = [name.strip() for name in args.datasets.split(",")]
        logger.info(f"Processing specific datasets: {dataset_names}")
    
    # Find datasets
    try:
        datasets = find_datasets(args.datasets_dir, dataset_names)
        if not datasets:
            logger.error("No datasets found to process")
            sys.exit(1)
        
        logger.info(f"Found {len(datasets)} dataset(s) to process:")
        for ds in datasets:
            logger.info(f"  - {ds.name}")
    
    except Exception as e:
        logger.error(f"Failed to find datasets: {e}")
        sys.exit(1)
    
    # Process each dataset with dynamic refresh
    results = {}
    processed_datasets = set()  # Track processed dataset names
    start_time = time.time()
    
    try:
        # Use a while loop to allow dynamic refresh of dataset list
        while True:
            # Refresh the dataset list from unfinished_datasets
            try:
                current_datasets = find_datasets(args.datasets_dir, dataset_names)
                # Filter out already processed datasets
                remaining_datasets = [
                    ds for ds in current_datasets 
                    if ds.name not in processed_datasets
                ]
            except Exception as e:
                logger.error(f"Failed to refresh dataset list: {e}")
                break
            
            # If no more datasets to process, exit the loop
            if not remaining_datasets:
                logger.info("No more datasets found in unfinished_datasets. Processing complete.")
                break
            
            logger.info(f"Found {len(remaining_datasets)} dataset(s) remaining to process:")
            for ds in remaining_datasets:
                logger.info(f"  - {ds.name}")
            
            # Process each remaining dataset
            for dataset_path in remaining_datasets:
                dataset_name = dataset_path.name
                
                # Skip if already processed (shouldn't happen, but safety check)
                if dataset_name in processed_datasets:
                    continue
                
                retries = 0
                success = False
                
                while retries <= args.max_retries and not success:
                    if retries > 0:
                        logger.info(f"Retry {retries}/{args.max_retries} for dataset {dataset_name}")
                        time.sleep(5)  # Brief pause before retry
                    
                    success = run_workflow_for_dataset(
                        dataset_path=dataset_path,
                        project_root=project_root,
                        workflow_type=args.workflow,
                        stages=args.stages,
                        verbose=args.verbose,
                        dry_run=args.dry_run,
                        logger=logger
                    )
                    
                    if success:
                        break
                    
                    retries += 1
                
                # Mark as processed
                processed_datasets.add(dataset_name)
                results[dataset_name] = success
                
                if success:
                    # Move completed dataset from unfinished_datasets to finished_datasets
                    unfinished_dir = args.datasets_dir
                    finished_dir = project_root / "datasets" / "finished_datasets"
                    
                    # Only move if we're reading from unfinished_datasets
                    if "unfinished_datasets" in str(unfinished_dir):
                        move_dataset_to_finished(
                            dataset_path=dataset_path,
                            unfinished_dir=unfinished_dir,
                            finished_dir=finished_dir,
                            logger=logger
                        )
                    
                    # After successful completion, break to refresh the list
                    logger.info("Refreshing dataset list to check for new datasets...")
                    break  # Break from inner for loop to refresh list
                else:
                    if args.continue_on_error:
                        logger.warning(f"Dataset {dataset_name} failed, continuing to next dataset...")
                        # Continue to next dataset in current list
                        continue
                    else:
                        logger.error(f"Dataset {dataset_name} failed, stopping batch processing")
                        sys.exit(1)
    
    except KeyboardInterrupt:
        logger.warning("⚠️ Batch processing interrupted by user")
        sys.exit(1)
    
    # Summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info("Batch processing summary:")
    logger.info(f"{'='*80}")
    
    successful = sum(1 for s in results.values() if s)
    failed = len(results) - successful
    
    logger.info(f"Total datasets processed: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    for dataset_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"  {dataset_name}: {status}")
    
    if failed > 0:
        sys.exit(1)
    else:
        logger.info("\n🎉 All datasets processed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

