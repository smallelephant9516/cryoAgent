"""RELION tools for cryoEM image processing."""

import os
import time
import subprocess
import shutil
import threading
import signal
from typing import Dict, Any, Optional, List
from pathlib import Path
from ..config.config_loader import RELIONSettings


class RELIONTools:
    """Tools for interacting with RELION."""
    
    def __init__(self, settings: RELIONSettings, config_loader=None):
        """Initialize RELION tools with connection settings."""
        self.settings = settings
        self.relion_exe = settings.relion_exe
        self.relion_dir = settings.relion_dir
        self.continue_job = settings.continue_job
        self.config_loader = config_loader
        
        # Ensure RELION directory exists
        os.makedirs(self.relion_dir, exist_ok=True)
        
        # Cache job metadata for monitoring
        self._job_cache: Dict[str, Dict[str, Any]] = {}
        
        # Backend execution settings
        self._backend_processes: Dict[str, subprocess.Popen] = {}
        self._backend_threads: Dict[str, threading.Thread] = {}
        self._backend_enabled = False
        
        # Initialize backend execution from settings
        if hasattr(settings, 'backend_execution') and settings.backend_execution:
            self._backend_enabled = settings.backend_execution.enabled
            self._backend_timeout = settings.backend_execution.default_timeout
            self._backend_check_interval = settings.backend_execution.check_interval
            self._max_concurrent_jobs = settings.backend_execution.max_concurrent_jobs
            self._auto_cleanup = settings.backend_execution.auto_cleanup
        else:
            self._backend_timeout = 3600
            self._backend_check_interval = 30
            self._max_concurrent_jobs = 3
            self._auto_cleanup = True
        
        # Verify RELION installation
        self._verify_relion_installation()
    
    def _get_workflow_params(self, step_name: str) -> Dict[str, Any]:
        """Get workflow parameters for a specific step from the configuration."""
        if not self.config_loader:
            return {}
        
        try:
            # Load the preprocessing config
            config = self.config_loader.load_config()
            # Access the workflow parameters from the merged config
            # This would need to be implemented based on how the config is structured
            return {}
        except Exception:
            return {}
    
    def _convert_to_relative_path(self, absolute_path: str) -> str:
        """
        Convert an absolute path to a relative path within the RELION working directory.
        Creates symbolic links if necessary.
        
        Args:
            absolute_path: Absolute path to convert
            
        Returns:
            Relative path within the RELION working directory
        """
        try:
            # Check if the absolute path exists
            if not os.path.exists(absolute_path):
                raise FileNotFoundError(f"Path does not exist: {absolute_path}")
            
            # Get the filename or directory name
            path_obj = Path(absolute_path)
            if path_obj.is_file():
                # For files, create a symbolic link with the filename
                link_name = path_obj.name
                link_path = os.path.join(self.relion_dir, link_name)
            else:
                # For directories, create a symbolic link with the directory name
                link_name = path_obj.name
                link_path = os.path.join(self.relion_dir, link_name)
            
            # Remove existing link if it exists
            if os.path.exists(link_path) or os.path.islink(link_path):
                os.unlink(link_path)
            
            # Create symbolic link
            os.symlink(absolute_path, link_path)
            
            print(f"Created symbolic link: {link_path} -> {absolute_path}")
            return link_name
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert path {absolute_path} to relative path: {e}")
    
    def _convert_movies_path_to_relative(self, movies_path: str) -> str:
        """
        Convert movies path (which may contain wildcards) to relative paths.
        
        Args:
            movies_path: Path pattern for movies (may contain wildcards)
            
        Returns:
            Relative path pattern for movies
        """
        import glob
        
        # If it's a glob pattern, handle it differently
        if "*" in movies_path:
            # Get the directory part of the path
            dir_path = os.path.dirname(movies_path)
            filename_pattern = os.path.basename(movies_path)
            
            # Convert the directory to relative path
            relative_dir = self._convert_to_relative_path(dir_path)
            
            # Return the relative path pattern
            return os.path.join(relative_dir, filename_pattern)
        else:
            # For non-glob paths, convert the entire path
            return self._convert_to_relative_path(movies_path)
    
    def _cleanup_old_links(self):
        """Clean up old symbolic links that might cause path conflicts."""
        try:
            # Remove old links that might conflict
            old_links = ['10025_subset', 'agent_test']
            for link_name in old_links:
                link_path = os.path.join(self.relion_dir, link_name)
                if os.path.exists(link_path) or os.path.islink(link_path):
                    os.unlink(link_path)
                    print(f"Removed old link: {link_path}")
        except Exception as e:
            print(f"Warning: Could not clean up old links: {e}")
    
    def _verify_relion_installation(self) -> None:
        """Verify RELION installation and accessibility."""
        try:
            # Check if RELION executable exists
            if not os.path.exists(self.relion_exe):
                raise FileNotFoundError(f"RELION executable not found at {self.relion_exe}")
            
            # Set environment variables to avoid display issues
            env = os.environ.copy()
            env['DISPLAY'] = ''
            env['QT_QPA_PLATFORM'] = 'offscreen'
            env['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
            env['QT_SCALE_FACTOR'] = '1'
            
            # Test RELION command
            result = subprocess.run(
                [self.relion_exe, "--version"],
                capture_output=True,
                text=True,
                timeout=30,
                env=env
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"RELION version check failed: {result.stderr}")
            
            print(f"✅ RELION installation verified: {result.stdout.strip()}")
            
        except Exception as e:
            raise ConnectionError(f"Failed to verify RELION installation: {e}")
    
    def import_movies(
        self,
        movies_path: str,
        output_dir: str,
        optics_group_name: str = "opticsGroup1",
        angpix: float = 0.6575,
        voltage: float = 300.0,
        cs: float = 2.7,
        q0: float = 0.1,
        beamtilt_x: float = 0.0,
        beamtilt_y: float = 0.0,
        output_file: str = "movies.star",
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        use_backend: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Import movies into RELION.
        
        Args:
            movies_path: Path to movie files (glob pattern)
            output_dir: Output directory for the job
            optics_group_name: Name for the optics group
            angpix: Pixel size in Angstroms
            voltage: Acceleration voltage in kV
            cs: Spherical aberration in mm
            q0: Amplitude contrast
            beamtilt_x: Beam tilt X in mrad
            beamtilt_y: Beam tilt Y in mrad
            output_file: Output STAR file name
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            check_interval: Time between status checks in seconds
            use_backend: Whether to run in backend mode
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing job information
        """
        try:
            # Clean up old links that might cause conflicts
            self._cleanup_old_links()
            
            # Create output directory
            full_output_dir = os.path.join(self.relion_dir, output_dir)
            os.makedirs(full_output_dir, exist_ok=True)
            
            # If backend execution is requested, use the backend method
            if use_backend:
                if not self._backend_enabled:
                    raise RuntimeError("Backend execution is not enabled. Call enable_backend_execution(True) first.")
                
                # Prepare command for backend execution
                cmd = self._prepare_import_movies_command(
                    movies_path, output_dir, optics_group_name, angpix, voltage, cs, q0,
                    beamtilt_x, beamtilt_y, output_file, **kwargs
                )
                
                # Generate job ID
                job_id = f"import_{output_dir}_{int(time.time())}"
                
                # Run in backend
                return self.run_relion_backend(
                    command=cmd,
                    job_id=job_id,
                    output_dir=output_dir,
                    timeout=timeout,
                    check_interval=check_interval,
                    **kwargs
                )
            
            # Convert absolute path to relative path if needed
            #movies_path = os.path.dirname(movies_path)
            if os.path.isabs(movies_path):
                print(f"Converting absolute path to relative: {movies_path}")
                relative_movies_path = self._convert_movies_path_to_relative(movies_path)
                print(f"Using relative path: {relative_movies_path}")
            else:
                relative_movies_path = movies_path
            output_dir_with_slash = output_dir+'/'
            
            # Prepare command
            cmd = [
                "relion_import",
                "--do_movies",
                f"--optics_group_name", optics_group_name,
                f"--angpix", str(angpix),
                f"--kV", str(voltage),
                f"--Cs", str(cs),
                f"--Q0", str(q0),
                f"--beamtilt_x", str(beamtilt_x),
                f"--beamtilt_y", str(beamtilt_y),
                f"--i", relative_movies_path,
                f"--odir", output_dir_with_slash,
                f"--ofile", output_file,
                "--continue",
                f"--pipeline_control", output_dir_with_slash
            ]
            
            # Add additional parameters from kwargs
            for key, value in kwargs.items():
                if value is not None:
                    cmd.extend([f"--{key}", str(value)])
            
            print(f"Running RELION import movies command: {' '.join(cmd)}")
            
            # Set environment variables to avoid display issues
            env = os.environ.copy()
            env['DISPLAY'] = ''
            env['QT_QPA_PLATFORM'] = 'offscreen'
            env['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
            env['QT_SCALE_FACTOR'] = '1'
            
            # Run the command from the RELION directory
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=self.relion_dir
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"RELION import movies failed: {result.stderr}")
            
            job_info = {
                "job_type": "relion_import_movies",
                "status": "completed",
                "output_dir": full_output_dir,
                "output_file": os.path.join(full_output_dir, output_file),
                "command": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            self._job_cache[f"import_{output_dir}"] = job_info
            
            print(f"✅ RELION import movies completed successfully!")
            print(f"Output file: {job_info['output_file']}")
            
            return job_info
            
        except Exception as e:
            raise RuntimeError(f"Failed to import movies: {e}")
    
    def motion_correction(
        self,
        input_star: str,
        output_dir: str,
        first_frame_sum: int = 1,
        last_frame_sum: int = -1,
        use_own: bool = True,
        use_motioncor2: bool = False,
        motioncor2_exe: Optional[str] = None,
        num_threads: int = 14,
        bin_factor: int = 1,
        bfactor: float = 150.0,
        dose_per_frame: float = 1.39,
        preexposure: float = 0.0,
        patch_x: int = 1,
        patch_y: int = 1,
        eer_grouping: int = 32,
        gainref: Optional[str] = None,
        gain_rot: int = 0,
        gain_flip: int = 0,
        dose_weighting: bool = True,
        grouping_for_ps: int = 3,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform motion correction on imported movies.
        
        Args:
            input_star: Path to input STAR file
            output_dir: Output directory for the job
            first_frame_sum: First frame to sum (default: 1)
            last_frame_sum: Last frame to sum (default: -1 for all)
            use_motioncor2: Use MotionCor2 instead of RELION's own implementation (default: False)
            motioncor2_exe: Path to MotionCor2 executable (default: None, uses environment variable)
            num_threads: Number of threads to use
            bin_factor: Binning factor
            bfactor: B-factor for dose weighting
            dose_per_frame: Dose per frame in e-/Å²
            preexposure: Preexposure dose
            patch_x: Patch size in X
            patch_y: Patch size in Y
            eer_grouping: EER grouping
            gainref: Path to gain reference file
            gain_rot: Gain rotation
            gain_flip: Gain flip
            dose_weighting: Enable dose weighting
            grouping_for_ps: Grouping for power spectrum
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            check_interval: Time between status checks in seconds
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing job information
        """
        try:
            # Create output directory
            full_output_dir = os.path.join(self.relion_dir, output_dir)
            os.makedirs(full_output_dir, exist_ok=True)
            
            # Prepare command
            cmd = [
                "which", "relion_run_motioncorr"
            ]
            
            # Get the full path to relion_run_motioncorr
            motioncorr_path = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            ).stdout.strip()
            
            if not motioncorr_path:
                raise RuntimeError("relion_run_motioncorr not found in PATH")
            
            output_dir_with_slash = output_dir+'/'
            
            cmd = [
                motioncorr_path,
                f"--i", input_star,
                f"--o", output_dir_with_slash,
                f"--first_frame_sum", str(first_frame_sum),
                f"--last_frame_sum", str(last_frame_sum),
                f"--j", str(num_threads),
                f"--bin_factor", str(bin_factor),
                f"--bfactor", str(bfactor),
                f"--dose_per_frame", str(dose_per_frame),
                f"--preexposure", str(preexposure),
                f"--patch_x", str(patch_x),
                f"--patch_y", str(patch_y),
                f"--eer_grouping", str(eer_grouping),
                f"--gain_rot", str(gain_rot),
                f"--gain_flip", str(gain_flip),
                f"--grouping_for_ps", str(grouping_for_ps),
                f"--pipeline_control", output_dir_with_slash
            ]
            
            if use_motioncor2:
                cmd.append("--use_motioncor2")
                if motioncor2_exe:
                    cmd.extend(["--motioncor2_exe", motioncor2_exe])
            else:
                cmd.append("--use_own")
            
            if gainref:
                # Convert absolute path to relative path if needed
                if os.path.isabs(gainref):
                    print(f"Converting gain reference absolute path to relative: {gainref}")
                    relative_gainref = self._convert_to_relative_path(gainref)
                    print(f"Using relative gain reference path: {relative_gainref}")
                else:
                    relative_gainref = gainref
                cmd.extend(["--gainref", relative_gainref])
            
            if dose_weighting:
                cmd.append("--dose_weighting")
            
            # Add additional parameters from kwargs
            for key, value in kwargs.items():
                if value is not None:
                    cmd.extend([f"--{key}", str(value)])
            
            print(f"Running RELION motion correction command: {' '.join(cmd)}")
            
            # Set environment variables to avoid display issues
            env = os.environ.copy()
            env['DISPLAY'] = ''
            env['QT_QPA_PLATFORM'] = 'offscreen'
            env['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
            env['QT_SCALE_FACTOR'] = '1'
            
            # Run the command from the RELION directory
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=self.relion_dir
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"RELION motion correction failed: {result.stderr}")
            
            job_info = {
                "job_type": "relion_motion_correction",
                "status": "completed",
                "output_dir": full_output_dir,
                "input_star": input_star,
                "command": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            self._job_cache[f"motioncorr_{output_dir}"] = job_info
            
            print(f"✅ RELION motion correction completed successfully!")
            print(f"Output directory: {full_output_dir}")
            
            return job_info
            
        except Exception as e:
            raise RuntimeError(f"Failed to run motion correction: {e}")
    
    def ctf_estimation(
        self,
        input_star: str,
        output_dir: str,
        box_size: int = 512,
        res_min: float = 30.0,
        res_max: float = 5.0,
        df_min: float = 5000.0,
        df_max: float = 50000.0,
        fstep: float = 500.0,
        dast: float = 100.0,
        ctffind_exe: str = "",
        ctf_win: int = -1,
        is_ctffind4: bool = True,
        fast_search: bool = True,
        only_do_unfinished: bool = True,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Estimate CTF parameters for micrographs.
        
        Args:
            input_star: Path to input STAR file
            output_dir: Output directory for the job
            box_size: Box size for CTF estimation
            res_min: Minimum resolution in Angstroms
            res_max: Maximum resolution in Angstroms
            df_min: Minimum defocus in Angstroms
            df_max: Maximum defocus in Angstroms
            fstep: Defocus step in Angstroms
            dast: Astigmatism step in Angstroms
            ctffind_exe: Path to CTFFIND executable
            ctf_win: CTF window size
            is_ctffind4: Use CTFFIND4
            fast_search: Enable fast search
            only_do_unfinished: Only process unfinished micrographs
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            check_interval: Time between status checks in seconds
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing job information
        """
        try:
            # Create output directory
            full_output_dir = os.path.join(self.relion_dir, output_dir)
            os.makedirs(full_output_dir, exist_ok=True)
            
            # Prepare command
            cmd = [
                "which", "relion_run_ctffind"
            ]
            
            # Get the full path to relion_run_ctffind
            ctffind_path = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            ).stdout.strip()
            
            if not ctffind_path:
                raise RuntimeError("relion_run_ctffind not found in PATH")

            output_dir_with_slash = output_dir+'/'
            
            cmd = [
                ctffind_path,
                f"--i", input_star,
                f"--o", output_dir_with_slash,
                f"--Box", str(box_size),
                f"--ResMin", str(res_min),
                f"--ResMax", str(res_max),
                f"--dFMin", str(df_min),
                f"--dFMax", str(df_max),
                f"--FStep", str(fstep),
                f"--dAst", str(dast),
                f"--ctffind_exe", ctffind_exe,
                f"--ctfWin", str(ctf_win),
                f"--pipeline_control", output_dir_with_slash
            ]
            
            if is_ctffind4:
                cmd.append("--is_ctffind4")
            
            if fast_search:
                cmd.append("--fast_search")
            
            if only_do_unfinished:
                cmd.append("--only_do_unfinished")
            
            # Add additional parameters from kwargs
            for key, value in kwargs.items():
                if value is not None:
                    cmd.extend([f"--{key}", str(value)])
            
            print(f"Running RELION CTF estimation command: {' '.join(cmd)}")
            
            # Set environment variables to avoid display issues
            env = os.environ.copy()
            env['DISPLAY'] = ''
            env['QT_QPA_PLATFORM'] = 'offscreen'
            env['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
            env['QT_SCALE_FACTOR'] = '1'
            
            # Run the command from the RELION directory
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=self.relion_dir
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"RELION CTF estimation failed: {result.stderr}")
            
            job_info = {
                "job_type": "relion_ctf_estimation",
                "status": "completed",
                "output_dir": full_output_dir,
                "input_star": input_star,
                "command": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            self._job_cache[f"ctffind_{output_dir}"] = job_info
            
            print(f"✅ RELION CTF estimation completed successfully!")
            print(f"Output directory: {full_output_dir}")
            
            return job_info
            
        except Exception as e:
            raise RuntimeError(f"Failed to run CTF estimation: {e}")
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a RELION job.
        
        Args:
            job_id: ID of the job to check
            
        Returns:
            Dictionary containing job status information
        """
        try:
            cached = self._job_cache.get(job_id, {})
            
            # If not in cache, try to find the job directory
            if not cached:
                # Try to construct the output directory path
                # Job IDs are typically in format "Import/job001", "MotionCorr/job002", etc.
                if "/" in job_id:
                    output_dir = os.path.join(self.relion_dir, job_id)
                else:
                    output_dir = os.path.join(self.relion_dir, job_id)
                
                if os.path.exists(output_dir):
                    cached = {
                        "output_dir": output_dir,
                        "status": "unknown",
                        "job_type": "unknown"
                    }
                else:
                    return {
                        "job_id": job_id,
                        "status": "unknown",
                        "message": "Job not found in cache and output directory not found"
                    }
            
            # Check if output files exist and get current status
            output_dir = cached.get("output_dir")
            if output_dir:
                # Check for completion indicators using helper method
                current_status = self._check_job_completion_files(output_dir)
                cached["status"] = current_status
                
                # Update cache with current status
                self._job_cache[job_id] = cached
            else:
                cached["status"] = "unknown"
            
            return {
                "job_id": job_id,
                "status": cached.get("status", "unknown"),
                "output_dir": cached.get("output_dir"),
                "job_type": cached.get("job_type"),
                "created_at": cached.get("created_at"),
                "updated_at": time.time()
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get job status for {job_id}: {e}")
    
    def _check_job_completion_files(self, output_dir: str) -> str:
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
    
    def is_job_completed(self, job_id: str) -> bool:
        """
        Check if a RELION job has completed successfully by looking for RELION_JOB_EXIT_SUCCESS.
        
        Args:
            job_id: ID of the job to check
            
        Returns:
            True if job completed successfully, False otherwise
        """
        try:
            status = self.get_job_status(job_id)
            return status.get("status") == "completed"
        except Exception:
            return False
    
    def wait_for_job_completion(
        self,
        job_id: str,
        timeout: int = 3600,
        check_interval: int = 30
    ) -> Dict[str, Any]:
        """
        Wait for a job to complete.
        
        Args:
            job_id: ID of the job to wait for
            timeout: Maximum time to wait in seconds
            check_interval: Time between status checks in seconds
            
        Returns:
            Final job status
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)
            
            if status["status"] in ["completed", "failed"]:
                return status
            
            print(f"Job {job_id} status: {status['status']}")
            time.sleep(check_interval)
        
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
    
    def monitor_job(
        self,
        job_id: str,
        timeout: int = 3600,
        check_interval: int = 30
    ) -> Dict[str, Any]:
        """Monitor a RELION job until it finishes or times out."""
        print(
            f"🛰️ Monitoring job {job_id} "
            f"(timeout={timeout}s, interval={check_interval}s)"
        )
        try:
            return self.wait_for_job_completion(
                job_id,
                timeout,
                check_interval
            )
        except TimeoutError:
            print(f"⏰ Job {job_id} timed out after {timeout} seconds")
            raise
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all cached jobs."""
        try:
            jobs = []
            for job_id, job_info in self._job_cache.items():
                jobs.append({
                    "job_id": job_id,
                    "job_type": job_info.get("job_type"),
                    "status": job_info.get("status"),
                    "output_dir": job_info.get("output_dir"),
                    "created_at": job_info.get("created_at")
                })
            return jobs
        except Exception as e:
            raise RuntimeError(f"Failed to list jobs: {e}")
    
    def get_job_output_directory(self, job_id: str) -> Dict[str, Any]:
        """
        Get the output directory and related information for a job.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Dictionary containing job directory information
        """
        try:
            cached = self._job_cache.get(job_id, {})
            
            if not cached:
                return {
                    "job_id": job_id,
                    "error": "Job not found in cache"
                }
            
            output_dir = cached.get("output_dir")
            
            if output_dir and os.path.exists(output_dir):
                # List files in output directory
                files = []
                for file in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        files.append({
                            "name": file,
                            "size": os.path.getsize(file_path),
                            "modified": os.path.getmtime(file_path)
                        })
                
                return {
                    "job_id": job_id,
                    "job_type": cached.get("job_type"),
                    "status": cached.get("status"),
                    "output_directory": output_dir,
                    "files": files
                }
            else:
                return {
                    "job_id": job_id,
                    "job_type": cached.get("job_type"),
                    "status": cached.get("status"),
                    "output_directory": output_dir,
                    "error": "Output directory not found"
                }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get job output directory for {job_id}: {e}")
    
    def get_job_log(self, job_id: str) -> Dict[str, Any]:
        """
        Read the log file of a RELION job to analyze errors and failures.
        
        Args:
            job_id: ID of the job to read logs for
            
        Returns:
            Dictionary containing log content and analysis
        """
        try:
            cached = self._job_cache.get(job_id, {})
            
            if not cached:
                return {
                    "success": False,
                    "error": "Job not found in cache",
                    "message": f"Could not find job {job_id}"
                }
            
            output_dir = cached.get("output_dir")
            if not output_dir or not os.path.exists(output_dir):
                return {
                    "success": False,
                    "error": "Output directory not found",
                    "message": f"Output directory for job {job_id} not found"
                }
            
            # Look for log files in the output directory
            log_files = []
            for file in os.listdir(output_dir):
                if file.endswith('.log') or file.endswith('.out') or file.endswith('.err'):
                    log_files.append(file)
            
            if not log_files:
                return {
                    "success": False,
                    "error": "No log files found",
                    "message": f"No log files found in {output_dir}"
                }
            
            # Read the first log file found
            log_file_path = os.path.join(output_dir, log_files[0])
            
            try:
                with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    log_content = f.read()
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error reading log file: {str(e)}",
                    "message": f"Failed to read log file: {log_file_path}"
                }
            
            # Analyze the log for common error patterns
            error_analysis = self._analyze_job_log(log_content)
            
            return {
                "success": True,
                "job_id": job_id,
                "log_file_path": log_file_path,
                "log_content": log_content,
                "log_size": len(log_content),
                "error_analysis": error_analysis,
                "message": f"Successfully read log for job {job_id}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to read job log for {job_id}: {str(e)}"
            }
    
    def _analyze_job_log(self, log_content: str) -> Dict[str, Any]:
        """
        Analyze job log content for common error patterns and provide insights.
        
        Args:
            log_content: Raw log content from the job
            
        Returns:
            Dictionary with error analysis and suggestions
        """
        analysis = {
            "has_errors": False,
            "error_types": [],
            "suggestions": [],
            "critical_errors": [],
            "warnings": [],
            "summary": ""
        }
        
        # Common error patterns to look for
        error_patterns = {
            "memory_error": [
                "out of memory", "memory error", "insufficient memory",
                "memory allocation failed", "oom", "killed"
            ],
            "parameter_error": [
                "invalid parameter", "parameter error", "bad parameter",
                "invalid value", "parameter out of range"
            ],
            "file_error": [
                "file not found", "no such file", "permission denied",
                "access denied", "file error", "path not found"
            ],
            "convergence_error": [
                "failed to converge", "convergence failed", "did not converge",
                "convergence error", "optimization failed"
            ],
            "ctf_error": [
                "ctf error", "ctf failed", "ctf estimation failed",
                "ctffind error", "ctf fit failed"
            ],
            "motion_correction_error": [
                "motion correction failed", "motioncorr error",
                "motion estimation failed", "drift correction failed"
            ]
        }
        
        # Check for error patterns
        log_lower = log_content.lower()
        for error_type, patterns in error_patterns.items():
            for pattern in patterns:
                if pattern in log_lower:
                    analysis["has_errors"] = True
                    if error_type not in analysis["error_types"]:
                        analysis["error_types"].append(error_type)
        
        # Generate suggestions based on error types
        if "memory_error" in analysis["error_types"]:
            analysis["suggestions"].append("Consider reducing the number of threads or using less memory")
            analysis["suggestions"].append("Try reducing the box size or using fewer particles")
        
        if "parameter_error" in analysis["error_types"]:
            analysis["suggestions"].append("Check parameter values and ranges")
            analysis["suggestions"].append("Verify input data format and compatibility")
        
        if "file_error" in analysis["error_types"]:
            analysis["suggestions"].append("Verify file paths and permissions")
            analysis["suggestions"].append("Check if input files exist and are accessible")
        
        if "ctf_error" in analysis["error_types"]:
            analysis["suggestions"].append("Check CTFFIND installation and parameters")
            analysis["suggestions"].append("Verify micrograph quality and defocus range")
        
        if "motion_correction_error" in analysis["error_types"]:
            analysis["suggestions"].append("Check motion correction parameters")
            analysis["suggestions"].append("Verify gain reference file and movie format")
        
        # Extract critical errors (lines containing ERROR)
        lines = log_content.split('\n')
        for line in lines:
            if 'error' in line.lower() and ('error:' in line.lower() or 'failed' in line.lower()):
                analysis["critical_errors"].append(line.strip())
        
        # Extract warnings
        for line in lines:
            if 'warning' in line.lower() or 'warn:' in line.lower():
                analysis["warnings"].append(line.strip())
        
        # Generate summary
        if analysis["has_errors"]:
            error_count = len(analysis["error_types"])
            analysis["summary"] = f"Job failed with {error_count} error type(s): {', '.join(analysis['error_types'])}"
        else:
            analysis["summary"] = "No obvious errors detected in log"
        
        return analysis
    
    def validate_inputs(self, input_type: str, input_path: str) -> str:
        """
        Validate input files and parameters for RELION processing.
        
        Args:
            input_type: Type of input ('movies', 'movie_files', 'star_file', 'files', etc.)
            input_path: Path to the input file or directory
            
        Returns:
            Validation result string
        """
        try:
            # Handle various input type names for movies
            if input_type in ["movies", "movie_files", "files"]:
                # Check if it's a glob pattern first
                if "*" in input_path:
                    import glob
                    files = glob.glob(input_path)
                    if not files:
                        # Try to find files in the directory part of the path
                        dir_path = os.path.dirname(input_path)
                        if os.path.exists(dir_path):
                            # Look for movie files in the directory
                            movie_files = [f for f in os.listdir(dir_path) if f.endswith(('.tif', '.mrc', '.mrcs'))]
                            if movie_files:
                                return f"✅ Found {len(movie_files)} movie files in directory: {dir_path} (pattern {input_path} didn't match, but files exist)"
                            else:
                                return f"❌ No movie files found in directory: {dir_path}"
                        else:
                            return f"❌ Directory does not exist: {dir_path}"
                    return f"✅ Found {len(files)} movie files matching pattern: {input_path}"
                
                # Check if it's a directory
                elif os.path.isdir(input_path):
                    files = [f for f in os.listdir(input_path) if f.endswith(('.tif', '.mrc', '.mrcs'))]
                    if not files:
                        return f"❌ No movie files found in directory: {input_path}"
                    return f"✅ Found {len(files)} movie files in directory: {input_path}"
                
                # Check if it's a single file
                elif os.path.exists(input_path):
                    if input_path.endswith(('.tif', '.mrc', '.mrcs')):
                        return f"✅ Movie file exists: {input_path}"
                    else:
                        return f"❌ File exists but is not a movie file: {input_path}"
                
                else:
                    return f"❌ Path does not exist: {input_path}"
            
            elif input_type in ["star_file", "star"]:
                if not os.path.exists(input_path):
                    return f"❌ Star file does not exist: {input_path}"
                return f"✅ Star file exists: {input_path}"
            
            else:
                return f"❌ Unknown input type: {input_type}. Supported types: movies, movie_files, files, star_file, star"
                
        except Exception as e:
            return f"❌ Error validating inputs: {str(e)}"
    
    def enable_backend_execution(self, enabled: bool = True) -> None:
        """
        Enable or disable backend execution for RELION commands.
        
        Args:
            enabled: Whether to enable backend execution
        """
        self._backend_enabled = enabled
        if enabled:
            print("✅ Backend execution enabled for RELION commands")
        else:
            print("❌ Backend execution disabled for RELION commands")
    
    def run_relion_backend(
        self,
        command: List[str],
        job_id: str,
        output_dir: str,
        timeout: Optional[int] = None,
        check_interval: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a RELION command in the background.
        
        Args:
            command: RELION command to execute
            job_id: Unique identifier for the job
            output_dir: Output directory for the job
            timeout: Maximum time to wait for completion in seconds (uses config default if None)
            check_interval: Time between status checks in seconds (uses config default if None)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing job information
        """
        if not self._backend_enabled:
            raise RuntimeError("Backend execution is not enabled. Call enable_backend_execution(True) first.")
        
        # Use configuration defaults if not provided
        if timeout is None:
            timeout = self._backend_timeout
        if check_interval is None:
            check_interval = self._backend_check_interval
        
        # Check concurrent job limit
        if len(self._backend_processes) >= self._max_concurrent_jobs:
            raise RuntimeError(f"Maximum concurrent backend jobs ({self._max_concurrent_jobs}) reached. Please wait for some jobs to complete.")
        
        try:
            # Create output directory
            full_output_dir = os.path.join(self.relion_dir, output_dir)
            os.makedirs(full_output_dir, exist_ok=True)
            
            # Set environment variables to avoid display issues
            env = os.environ.copy()
            env['DISPLAY'] = ''
            env['QT_QPA_PLATFORM'] = 'offscreen'
            env['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
            env['QT_SCALE_FACTOR'] = '1'
            
            # Start the process in the background
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=self.relion_dir,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Store process and create monitoring thread
            self._backend_processes[job_id] = process
            
            # Create monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitor_backend_process,
                args=(job_id, process, full_output_dir, timeout, check_interval),
                daemon=True
            )
            self._backend_threads[job_id] = monitor_thread
            monitor_thread.start()
            
            job_info = {
                "job_id": job_id,
                "job_type": "relion_backend",
                "status": "running",
                "output_dir": full_output_dir,
                "command": " ".join(command),
                "process_id": process.pid,
                "started_at": time.time()
            }
            
            self._job_cache[job_id] = job_info
            
            print(f"🚀 Started RELION backend job: {job_id}")
            print(f"   Process ID: {process.pid}")
            print(f"   Output directory: {full_output_dir}")
            print(f"   Command: {' '.join(command)}")
            
            return job_info
            
        except Exception as e:
            raise RuntimeError(f"Failed to start RELION backend job: {e}")
    
    def _monitor_backend_process(
        self,
        job_id: str,
        process: subprocess.Popen,
        output_dir: str,
        timeout: int,
        check_interval: int
    ) -> None:
        """
        Monitor a background RELION process.
        
        Args:
            job_id: Job identifier
            process: The subprocess to monitor
            output_dir: Output directory for the job
            timeout: Maximum time to wait for completion
            check_interval: Time between status checks
        """
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout:
                # Check if process is still running
                if process.poll() is not None:
                    # Process has finished
                    stdout, stderr = process.communicate()
                    
                    # Update job status
                    if process.returncode == 0:
                        status = "completed"
                        print(f"✅ Backend job {job_id} completed successfully")
                    else:
                        status = "failed"
                        print(f"❌ Backend job {job_id} failed with return code {process.returncode}")
                        print(f"   Error: {stderr}")
                    
                    # Update job cache
                    if job_id in self._job_cache:
                        self._job_cache[job_id].update({
                            "status": status,
                            "return_code": process.returncode,
                            "stdout": stdout,
                            "stderr": stderr,
                            "completed_at": time.time()
                        })
                    
                    break
                
                # Check for completion files
                completion_status = self._check_job_completion_files(output_dir)
                if completion_status in ["completed", "failed"]:
                    if job_id in self._job_cache:
                        self._job_cache[job_id]["status"] = completion_status
                    break
                
                time.sleep(check_interval)
            
            else:
                # Timeout reached
                print(f"⏰ Backend job {job_id} timed out after {timeout} seconds")
                if job_id in self._job_cache:
                    self._job_cache[job_id]["status"] = "timeout"
                
                # Terminate the process
                self._terminate_backend_process(job_id)
        
        except Exception as e:
            print(f"❌ Error monitoring backend job {job_id}: {e}")
            if job_id in self._job_cache:
                self._job_cache[job_id]["status"] = "error"
                self._job_cache[job_id]["error"] = str(e)
        
        finally:
            # Clean up
            if job_id in self._backend_processes:
                del self._backend_processes[job_id]
            if job_id in self._backend_threads:
                del self._backend_threads[job_id]
    
    def _terminate_backend_process(self, job_id: str) -> None:
        """
        Terminate a background RELION process.
        
        Args:
            job_id: Job identifier to terminate
        """
        if job_id in self._backend_processes:
            process = self._backend_processes[job_id]
            try:
                if os.name != 'nt':
                    # On Unix-like systems, terminate the process group
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    # On Windows, terminate the process
                    process.terminate()
                
                # Wait a bit for graceful termination
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate gracefully
                    if os.name != 'nt':
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:
                        process.kill()
                
                print(f"🛑 Terminated backend job {job_id}")
                
            except Exception as e:
                print(f"❌ Error terminating backend job {job_id}: {e}")
    
    def get_backend_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a backend RELION job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dictionary containing job status information
        """
        if job_id not in self._job_cache:
            return {
                "job_id": job_id,
                "status": "not_found",
                "message": "Job not found"
            }
        
        job_info = self._job_cache[job_id].copy()
        
        # Check if process is still running
        if job_id in self._backend_processes:
            process = self._backend_processes[job_id]
            if process.poll() is None:
                job_info["status"] = "running"
                job_info["process_id"] = process.pid
            else:
                # Process has finished, get final status
                stdout, stderr = process.communicate()
                job_info.update({
                    "status": "completed" if process.returncode == 0 else "failed",
                    "return_code": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr
                })
        
        return job_info
    
    def list_backend_jobs(self) -> List[Dict[str, Any]]:
        """
        List all backend jobs.
        
        Returns:
            List of job information dictionaries
        """
        jobs = []
        for job_id, job_info in self._job_cache.items():
            if job_info.get("job_type") == "relion_backend":
                jobs.append({
                    "job_id": job_id,
                    "status": job_info.get("status", "unknown"),
                    "output_dir": job_info.get("output_dir"),
                    "process_id": job_info.get("process_id"),
                    "started_at": job_info.get("started_at"),
                    "completed_at": job_info.get("completed_at")
                })
        return jobs
    
    def stop_backend_job(self, job_id: str) -> bool:
        """
        Stop a running backend job.
        
        Args:
            job_id: Job identifier to stop
            
        Returns:
            True if job was stopped successfully, False otherwise
        """
        if job_id not in self._backend_processes:
            print(f"❌ Backend job {job_id} not found or not running")
            return False
        
        try:
            self._terminate_backend_process(job_id)
            if job_id in self._job_cache:
                self._job_cache[job_id]["status"] = "stopped"
            return True
        except Exception as e:
            print(f"❌ Error stopping backend job {job_id}: {e}")
            return False
    
    def stop_all_backend_jobs(self) -> int:
        """
        Stop all running backend jobs.
        
        Returns:
            Number of jobs stopped
        """
        stopped_count = 0
        for job_id in list(self._backend_processes.keys()):
            if self.stop_backend_job(job_id):
                stopped_count += 1
        return stopped_count
    
    def _prepare_import_movies_command(
        self,
        movies_path: str,
        output_dir: str,
        optics_group_name: str,
        angpix: float,
        voltage: float,
        cs: float,
        q0: float,
        beamtilt_x: float,
        beamtilt_y: float,
        output_file: str,
        **kwargs
    ) -> List[str]:
        """
        Prepare the RELION import movies command.
        
        Args:
            movies_path: Path to movie files
            output_dir: Output directory
            optics_group_name: Optics group name
            angpix: Pixel size
            voltage: Acceleration voltage
            cs: Spherical aberration
            q0: Amplitude contrast
            beamtilt_x: Beam tilt X
            beamtilt_y: Beam tilt Y
            output_file: Output file name
            **kwargs: Additional parameters
            
        Returns:
            Command as list of strings
        """
        # Convert absolute path to relative path if needed
        if os.path.isabs(movies_path):
            print(f"Converting absolute path to relative: {movies_path}")
            relative_movies_path = self._convert_movies_path_to_relative(movies_path)
            print(f"Using relative path: {relative_movies_path}")
        else:
            relative_movies_path = movies_path
        
        output_dir_with_slash = output_dir + '/'
        
        # Prepare command
        cmd = [
            "relion_import",
            "--do_movies",
            f"--optics_group_name", optics_group_name,
            f"--angpix", str(angpix),
            f"--kV", str(voltage),
            f"--Cs", str(cs),
            f"--Q0", str(q0),
            f"--beamtilt_x", str(beamtilt_x),
            f"--beamtilt_y", str(beamtilt_y),
            f"--i", relative_movies_path,
            f"--odir", output_dir_with_slash,
            f"--ofile", output_file,
            "--continue",
            f"--pipeline_control", output_dir_with_slash
        ]
        
        # Add additional parameters from kwargs
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key}", str(value)])
        
        return cmd