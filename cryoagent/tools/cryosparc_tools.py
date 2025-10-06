"""CryoSPARC tools for cryoEM image processing."""

import time
from typing import Dict, Any, Optional, List
from cryosparc.tools import CryoSPARC
from ..config.config_loader import CryoSPARCSettings


class CryoSPARCTools:
    """Tools for interacting with CryoSPARC."""
    
    def __init__(self, settings: CryoSPARCSettings):
        """Initialize CryoSPARC tools with connection settings."""
        self.settings = settings
        self.cs = None
        # Cache job metadata so we can resolve project/workspace during monitoring
        self._job_cache: Dict[str, Dict[str, Optional[str]]] = {}
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to CryoSPARC."""
        try:
            # Try different connection parameter combinations
            connection_attempts = []
            
            # Attempt 1: With license and email (most common case)
            if (self.settings.license_id and self.settings.license_id != "your-cryosparc-license-id-here" and
                self.settings.username and self.settings.username != "your-username"):
                connection_attempts.append({
                    "host": self.settings.host,
                    "base_port": self.settings.base_port,
                    "license": self.settings.license_id,
                    "email": self.settings.username,
                    "password": self.settings.password
                })
            
            # Attempt 2: With license and email only (no password)
            if (self.settings.license_id and self.settings.license_id != "your-cryosparc-license-id-here" and
                self.settings.username and self.settings.username != "your-username"):
                connection_attempts.append({
                    "host": self.settings.host,
                    "base_port": self.settings.base_port,
                    "license": self.settings.license_id,
                    "email": self.settings.username
                })
            
            # Attempt 3: Basic connection with host and port only (fallback)
            connection_attempts.append({
                "host": self.settings.host,
                "base_port": self.settings.base_port
            })
            
            # Try each connection attempt
            last_error = None
            for i, params in enumerate(connection_attempts):
                try:
                    print(f"Attempting CryoSPARC connection {i+1}/{len(connection_attempts)} with params: {list(params.keys())}")
                    self.cs = CryoSPARC(**params)
                    assert self.cs.test_connection()
                    print(f"✅ Successfully connected to CryoSPARC at {self.settings.host}:{self.settings.base_port}")
                    return
                except Exception as e:
                    last_error = e
                    print(f"❌ Connection attempt {i+1} failed: {e}")
                    continue
            
            # If all attempts failed, raise the last error
            raise ConnectionError(f"All CryoSPARC connection attempts failed. Last error: {last_error}")
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to CryoSPARC: {e}")
    
    def import_movies(
        self,
        project_uid: str,
        workspace_uid: str,
        movies_path: str,
        gain_ref_path: Optional[str] = None,
        pixel_size: float = 1.0,
        voltage: float = 300.0,
        cs_mm: float = 2.7,
        dose: float = 1.0,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Import movies into CryoSPARC.
        
        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            movies_path: Path to movie files
            gain_ref_path: Path to gain reference file
            pixel_size: Pixel size in Angstroms
            voltage: Acceleration voltage in kV
            cs_mm: Spherical aberration in mm
            dose: Electron dose per frame in e-/Å²
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing job information
        """
        try:
            # Find project and workspace
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)
            
            # Prepare job parameters using correct CryoSPARC API format
            job_params = {
                "blob_paths": movies_path,
                "psize_A": pixel_size,
                "accel_kv": voltage,
                "cs_mm": cs_mm,
                "total_dose_e_per_A2": dose,
                **kwargs
            }
            
            if gain_ref_path:
                job_params["gainref_path"] = gain_ref_path
            
            # Create job using workspace.create_job()
            job = workspace.create_job("import_movies", params=job_params)
            
            # Queue the job
            job.queue()
            print(f"Queued import movies job: {job.uid}")
            
            self._job_cache[job.uid] = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            result = {
                "job_uid": job.uid,
                "job_type": "import_movies",
                "status": "queued",
                "params": job_params,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            
            # Wait for completion if requested
            if wait_for_completion:
                print(f"⏳ Waiting for import movies job {job.uid} to complete...")
                try:
                    final_status = self.wait_for_job_completion(
                        project_uid,
                        job.uid,
                        workspace_uid,
                        timeout,
                        check_interval
                    )
                    result["status"] = final_status["status"]
                    result["final_status"] = final_status
                    if final_status["status"] == "completed":
                        print(f"✅ Import movies job {job.uid} completed successfully!")
                    else:
                        print(f"⚠️ Import movies job {job.uid} finished with status: {final_status['status']}")
                except TimeoutError:
                    result["status"] = "timeout"
                    print(f"⏰ Import movies job {job.uid} timed out after {timeout} seconds")
                except Exception as e:
                    result["status"] = "error"
                    print(f"❌ Error monitoring import movies job {job.uid}: {e}")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to import movies: {e}")
    
    def motion_correction(
        self,
        project_uid: str,
        workspace_uid: str,
        movies_job_uid: str,
        binning: int = 1,
        patch_size: int = 5,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform motion correction on imported movies.
        
        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            movies_job_uid: UID of the import movies job
            binning: Binning factor for motion correction
            patch_size: Patch size for motion correction
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing job information
        """
        try:
            # Find project and workspace
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)
            
            # Prepare job parameters using correct CryoSPARC API format
            job_params = {
                **kwargs
            }
            
            # Create job with connections - try known import job outputs for compatibility
            connection_errors = []
            job = None
            for output_label in ("imported_movies", "movies"):
                try:
                    job = workspace.create_job(
                        "patch_motion_correction_multi",
                        params=job_params,
                        connections={"movies": (movies_job_uid, output_label)}
                    )
                    break
                except Exception as exc:  # store and try next label
                    connection_errors.append((output_label, exc))
                    job = None
            if job is None:
                error_messages = ", ".join(
                    f"output '{label}': {err}" for label, err in connection_errors
                ) or "unknown"
                raise RuntimeError(
                    "Unable to connect motion correction to import job outputs: "
                    f"{error_messages}"
                )
            
            # Queue the job
            used_lane = lane
            try:
                job.queue(lane=lane, hostname=hostname)
            except Exception as queue_error:
                message = str(queue_error)
                if (lane is None and hostname is None and "Must specify a lane" in message):
                    try:
                        lanes = self.cs.get_lanes()
                        if not lanes:
                            raise queue_error
                        used_lane = lanes[0]["name"]
                        print(f"⚙️ No lane specified; retrying queue on lane '{used_lane}'")
                        job.queue(lane=used_lane)
                    except Exception:
                        raise queue_error
                else:
                    raise queue_error
            print(f"Queued motion correction job: {job.uid}")
            
            self._job_cache[job.uid] = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            result = {
                "job_uid": job.uid,
                "job_type": "patch_motion_correction_multi",
                "status": "queued",
                "params": job_params,
                "connections": {"movies": movies_job_uid},
                "lane": used_lane,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            
            # Wait for completion if requested
            if wait_for_completion:
                print(f"⏳ Waiting for motion correction job {job.uid} to complete...")
                try:
                    final_status = self.wait_for_job_completion(
                        project_uid,
                        job.uid,
                        workspace_uid,
                        timeout,
                        check_interval
                    )
                    result["status"] = final_status["status"]
                    result["final_status"] = final_status
                    if final_status["status"] == "completed":
                        print(f"✅ Motion correction job {job.uid} completed successfully!")
                    else:
                        print(f"⚠️ Motion correction job {job.uid} finished with status: {final_status['status']}")
                except TimeoutError:
                    result["status"] = "timeout"
                    print(f"⏰ Motion correction job {job.uid} timed out after {timeout} seconds")
                except Exception as e:
                    result["status"] = "error"
                    print(f"❌ Error monitoring motion correction job {job.uid}: {e}")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to start motion correction: {e}")
    
    def ctf_estimation(
        self,
        project_uid: str,
        workspace_uid: str,
        micrographs_job_uid: str,
        min_res: float = 30.0,
        max_res: float = 4.0,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Estimate CTF parameters for micrographs.
        
        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            micrographs_job_uid: UID of the motion correction job
            min_res: Minimum resolution for CTF estimation
            max_res: Maximum resolution for CTF estimation
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing job information
        """
        try:
            # Find project and workspace
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)
            
            # Prepare job parameters using correct CryoSPARC API format
            job_params = {
                **kwargs
            }
            
            # Create job with connections - use job UID directly for connections
            job = workspace.create_job(
                "patch_ctf_estimation_multi",
                params=job_params,
                connections={"exposures": (micrographs_job_uid, "micrographs")}
            )
            
            # Queue the job
            used_lane = lane
            try:
                job.queue(lane=lane, hostname=hostname)
            except Exception as queue_error:
                message = str(queue_error)
                if (lane is None and hostname is None and "Must specify a lane" in message):
                    try:
                        lanes = self.cs.get_lanes()
                        if not lanes:
                            raise queue_error
                        used_lane = lanes[0]["name"]
                        print(f"⚙️ No lane specified; retrying queue on lane '{used_lane}'")
                        job.queue(lane=used_lane)
                    except Exception:
                        raise queue_error
                else:
                    raise queue_error
            print(f"Queued CTF estimation job: {job.uid}")
            
            self._job_cache[job.uid] = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            result = {
                "job_uid": job.uid,
                "job_type": "patch_ctf_estimation_multi",
                "status": "queued",
                "params": job_params,
                "connections": {"exposures": micrographs_job_uid},
                "lane": used_lane,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            
            # Wait for completion if requested
            if wait_for_completion:
                print(f"⏳ Waiting for CTF estimation job {job.uid} to complete...")
                try:
                    final_status = self.wait_for_job_completion(
                        project_uid,
                        job.uid,
                        workspace_uid,
                        timeout,
                        check_interval
                    )
                    result["status"] = final_status["status"]
                    result["final_status"] = final_status
                    if final_status["status"] == "completed":
                        print(f"✅ CTF estimation job {job.uid} completed successfully!")
                    else:
                        print(f"⚠️ CTF estimation job {job.uid} finished with status: {final_status['status']}")
                except TimeoutError:
                    result["status"] = "timeout"
                    print(f"⏰ CTF estimation job {job.uid} timed out after {timeout} seconds")
                except Exception as e:
                    result["status"] = "error"
                    print(f"❌ Error monitoring CTF estimation job {job.uid}: {e}")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to start CTF estimation: {e}")
    
    def micrograph_selection(
        self,
        project_uid: str,
        workspace_uid: str,
        ctf_job_uid: str,
        min_resolution: float = 5.0,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Select micrographs with resolution better than specified threshold using curate_exposures_v2.
        
        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            ctf_job_uid: UID of the CTF estimation job
            min_resolution: Minimum resolution threshold in Angstroms (default: 5.0)
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing job information
        """
        try:
            # Find project and workspace
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)
            
            # Create curate_exposures_v2 job with proper connections
            job = workspace.create_job(
                "curate_exposures_v2",
                connections={"exposures": (ctf_job_uid, "exposures")},
                **kwargs
            )
            
            # Queue the job
            job.queue()
            print(f"Queued micrograph curation job: {job.uid}")
            
            # Wait for job to reach waiting status (interactive mode)
            job.wait_for_status("waiting")
            print(f"Job {job.uid} reached waiting status, configuring thresholds...")
            
            # Get fields and thresholds data
            data = job.interact("get_fields_and_thresholds")
            
            # Find the CTF resolution field and set threshold
            from cryosparc.util import first
            ctf_res_field = first(field for field in data["fields"] if field["name"] == "ctf_fit_to_A")
            
            if ctf_res_field:
                # Set threshold to filter micrographs with resolution better than min_resolution
                ctf_res_field["thresholds"] = [1, min_resolution]  # Keep micrographs with resolution 1 to min_resolution Å
                ctf_res_field["active"] = True
                print(f"Set CTF resolution threshold to {min_resolution} Å")
            else:
                print("⚠️ Warning: Could not find 'ctf_fit_to_A' field in CTF data")
            
            # Apply the thresholds
            job.interact("set_thresholds", data)
            job.interact("shutdown_interactive")
            
            self._job_cache[job.uid] = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            result = {
                "job_uid": job.uid,
                "job_type": "curate_exposures_v2",
                "status": "queued",
                "params": {
                    "ctf_job_uid": ctf_job_uid,
                    "min_resolution": min_resolution
                },
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            
            # Wait for completion if requested
            if wait_for_completion:
                print(f"⏳ Waiting for micrograph curation job {job.uid} to complete...")
                try:
                    final_status = self.wait_for_job_completion(
                        project_uid,
                        job.uid,
                        workspace_uid,
                        timeout,
                        check_interval
                    )
                    result["status"] = final_status["status"]
                    result["final_status"] = final_status
                    if final_status["status"] == "completed":
                        print(f"✅ Micrograph curation job {job.uid} completed successfully!")
                    else:
                        print(f"⚠️ Micrograph curation job {job.uid} finished with status: {final_status['status']}")
                except TimeoutError:
                    result["status"] = "timeout"
                    print(f"⏰ Micrograph curation job {job.uid} timed out after {timeout} seconds")
                except Exception as e:
                    result["status"] = "error"
                    print(f"❌ Error monitoring micrograph curation job {job.uid}: {e}")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to start micrograph selection: {e}")
    
    def get_job_status(
        self,
        job_uid: str,
        project_uid: Optional[str] = None,
        workspace_uid: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get the status of a CryoSPARC job.
        
        Args:
            job_uid: UID of the job to check
            project_uid: Optional project UID containing the job
            workspace_uid: Optional workspace UID containing the job
            
        Returns:
            Dictionary containing job status information
        """
        try:
            cached = self._job_cache.get(job_uid, {})
            project_uid = project_uid or cached.get("project_uid")
            workspace_uid = workspace_uid or cached.get("workspace_uid")

            if not project_uid:
                raise ValueError(
                    "Project UID is required to fetch job status. "
                    "Pass project_uid explicitly or ensure the job was queued via CryoSPARCTools."
                )

            job = self.cs.find_job(project_uid, job_uid)
            job.refresh()
            doc = getattr(job, "doc", {})
            status = doc.get("status", getattr(job, "status", "unknown"))

            # Attempt to extract progress/message from known fields, falling back safely
            progress = doc.get("meta", {}).get("progress")
            if progress is None:
                progress = doc.get("status_stream", {}).get("progress") if isinstance(doc.get("status_stream"), dict) else None
            message = (
                doc.get("status_string")
                or doc.get("log")
                or doc.get("meta", {}).get("status")
                or ""
            )

            return {
                "job_uid": job_uid,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "status": status,
                "progress": progress,
                "message": message,
                "created_at": doc.get("created_at"),
                "updated_at": doc.get("updated_at"),
                "job_type": doc.get("job_type")
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get job status for {job_uid}: {e}")
    
    def wait_for_job_completion(
        self,
        project_uid: str,
        job_uid: str,
        workspace_uid: Optional[str] = None,
        timeout: int = 3600,
        check_interval: int = 30
    ) -> Dict[str, Any]:
        """
        Wait for a job to complete.
        
        Args:
            job_uid: UID of the job to wait for
            timeout: Maximum time to wait in seconds
            check_interval: Time between status checks in seconds
            
        Returns:
            Final job status
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_job_status(
                job_uid,
                project_uid=project_uid,
                workspace_uid=workspace_uid
            )
            
            if status["status"] in ["completed", "failed", "cancelled"]:
                return status
            
            progress_display = f"{status['progress']}%" if status.get("progress") is not None else "N/A"
            message = status.get("message")
            line = f"Job {job_uid} status: {status['status']} ({progress_display})"
            if message:
                line += f" - {message}"
            print(line)
            time.sleep(check_interval)
        
        raise TimeoutError(f"Job {job_uid} did not complete within {timeout} seconds")

    def monitor_job(
        self,
        project_uid: str,
        job_uid: str,
        workspace_uid: Optional[str] = None,
        timeout: int = 3600,
        check_interval: int = 30
    ) -> Dict[str, Any]:
        """Monitor a CryoSPARC job until it finishes or times out."""
        print(
            f"🛰️ Monitoring job {job_uid} "
            f"(timeout={timeout}s, interval={check_interval}s)"
        )
        try:
            return self.wait_for_job_completion(
                project_uid,
                job_uid,
                workspace_uid,
                timeout,
                check_interval
            )
        except TimeoutError:
            print(f"⏰ Job {job_uid} timed out after {timeout} seconds")
            raise
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all available projects."""
        try:
            # Use the correct CryoSPARC API to list projects
            projects = []
            for project in self.cs.list_projects():
                projects.append({
                    "uid": project.uid,
                    "name": project.name,
                    "created_at": getattr(project, 'created_at', None),
                    "updated_at": getattr(project, 'updated_at', None)
                })
            return projects
        except Exception as e:
            raise RuntimeError(f"Failed to list projects: {e}")
    
    def list_workspaces(self, project_uid: str) -> List[Dict[str, Any]]:
        """List workspaces in a project."""
        try:
            # Use the correct CryoSPARC API to list workspaces
            project = self.cs.find_project(project_uid)
            workspaces = []
            for workspace in project.list_workspaces():
                workspaces.append({
                    "uid": workspace.uid,
                    "name": workspace.name,
                    "created_at": getattr(workspace, 'created_at', None),
                    "updated_at": getattr(workspace, 'updated_at', None)
                })
            return workspaces
        except Exception as e:
            raise RuntimeError(f"Failed to list workspaces for project {project_uid}: {e}")
