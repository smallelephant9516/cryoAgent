#!/usr/bin/env python3
"""
Batch Dataset Runner for CryoAgent

This script runs the CryoAgent workflow on multiple datasets continuously.
Each dataset should have its own folder in the datasets/ directory with:
- configs/session.json
- configs/microscope_config.json
- configs/master_config.json (can be a symlink to the main one)

Usage:
    python run_batch_datasets.py [options]

Options:
    --datasets-dir DIR       Directory containing dataset folders (default: datasets/)
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


def collect_outputs_after_timestamp(outputs_dir: Path, timestamp: float) -> List[Path]:
    """
    Collect all output files created after a given timestamp.
    
    Args:
        outputs_dir: Directory to search for outputs
        timestamp: Unix timestamp to compare against
        
    Returns:
        List of file paths created after the timestamp
    """
    if not outputs_dir.exists():
        return []
    
    new_files = []
    for file_path in outputs_dir.rglob("*"):
        if file_path.is_file():
            try:
                file_mtime = file_path.stat().st_mtime
                if file_mtime >= timestamp:
                    new_files.append(file_path)
            except (OSError, AttributeError):
                continue
    
    return sorted(new_files)


def move_outputs_to_dataset(output_files: List[Path], dataset_output_dir: Path, outputs_root: Path, logger: logging.Logger) -> None:
    """
    Move output files to the dataset-specific output directory.
    
    Args:
        output_files: List of output file paths to move
        dataset_output_dir: Target directory for dataset outputs
        outputs_root: Root outputs directory (to calculate relative paths)
        logger: Logger instance
    """
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    for src_file in output_files:
        try:
            # Calculate relative path from outputs root
            try:
                rel_path = src_file.relative_to(outputs_root)
                # If file is directly in outputs/, move it to dataset/outputs/
                # If file is in a subdirectory, preserve that structure
                if rel_path.parent == Path('.'):
                    dst_file = dataset_output_dir / rel_path.name
                else:
                    # Preserve subdirectory structure by creating it in dataset/outputs/
                    dst_file = dataset_output_dir / rel_path
            except ValueError:
                # If relative_to fails, just use the filename
                dst_file = dataset_output_dir / src_file.name
            
            # Create parent directories if needed
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            shutil.move(str(src_file), str(dst_file))
            logger.info(f"Moved output: {src_file.name} -> {dst_file.relative_to(dataset_output_dir.parent)}")
            
            # Also move parent directories if they become empty
            parent = src_file.parent
            while parent != outputs_root and parent.exists():
                try:
                    if not any(parent.iterdir()):  # Directory is empty
                        parent.rmdir()
                        logger.debug(f"Removed empty directory: {parent}")
                        parent = parent.parent
                    else:
                        break
                except OSError:
                    break
            
        except Exception as e:
            logger.warning(f"Failed to move output file {src_file}: {e}")


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
    
    # Record timestamp before workflow starts to identify new outputs
    outputs_dir = project_root / "outputs"
    timestamp_before = time.time()
    
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
        
        # Collect outputs created during this workflow run
        logger.info("Collecting outputs for dataset...")
        new_output_files = collect_outputs_after_timestamp(outputs_dir, timestamp_before)
        
        if new_output_files:
            logger.info(f"Found {len(new_output_files)} output file(s) to move to dataset folder")
            move_outputs_to_dataset(new_output_files, dataset_output_dir, outputs_dir, logger)
            logger.info(f"✅ Moved outputs to {dataset_output_dir}")
        else:
            logger.warning(f"No new output files found in {outputs_dir} after workflow run")
        
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
        default=project_root / "datasets",
        help="Directory containing dataset folders (default: datasets/)"
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
    
    # Process each dataset
    results = {}
    start_time = time.time()
    
    try:
        for dataset_path in datasets:
            dataset_name = dataset_path.name
            
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
            
            results[dataset_name] = success
            
            if not success:
                if args.continue_on_error:
                    logger.warning(f"Dataset {dataset_name} failed, continuing to next dataset...")
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

