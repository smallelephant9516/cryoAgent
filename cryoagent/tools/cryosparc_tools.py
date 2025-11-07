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
    
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_particles_output_slot(self, project, job_uid: str) -> str:
        """Infer the appropriate particles output slot for a CryoSPARC job."""
        default_slots = [
            "particles",
            "particles_selected",
            "imported_particles",
            "selected_particles",
            "particles_all_classes",
        ]

        try:
            job = project.find_job(job_uid)
            job.refresh()
            doc = getattr(job, "doc", {}) or {}
            outputs = doc.get("output_result_groups", []) or []

            for group in outputs:
                name = group.get("name") or ""
                group_type = (group.get("type") or "").lower()
                if "particle" in group_type or "particle" in name.lower():
                    return name

            job_type = (doc.get("type") or doc.get("job_type") or "").lower()
            if "import" in job_type:
                return "imported_particles"
            if "select" in job_type:
                return "particles_selected"
            if "extract" in job_type:
                return "particles"
        except Exception:
            pass

        for slot in default_slots:
            try:
                job = project.find_job(job_uid)
                job.refresh()
                doc = getattr(job, "doc", {}) or {}
                outputs = doc.get("output_result_groups", []) or []
                if any((group.get("name") or "") == slot for group in outputs):
                    return slot
            except Exception:
                continue

        return "particles"

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

    def import_particles_from_star(
        self,
        project_uid: str,
        workspace_uid: str,
        star_path: str,
        *,
        data_sign: Optional[str] = None,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 1800,
        check_interval: int = 30,
        **kwargs,
    ) -> Dict[str, Any]:
        """Import RELION particles (.star) into CryoSPARC using the Import Particles job.

        Tries multiple parameter keys for compatibility across CryoSPARC versions while keeping args minimal.
        """
        try:
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)

            # Try robust param key variants
            param_variants = [
                {"particle_meta_path": star_path},
                {"particles_star_path": star_path},
                {"star_path": star_path},
                {"particles_path": star_path},
                {"input_star_path": star_path},
            ]

            # Optional data sign parameter (CryoSPARC expects one of {"positive","negative"})
            if data_sign:
                ds = str(data_sign).lower()
                sign_key_variants = []
                if ds in {"positive", "+", "plus", "dark-on-light"}:
                    sign_key_variants.append(("sign", "dark-on-light"))
                elif ds in {"negative", "-", "minus", "light-on-dark"}:
                    sign_key_variants.append(("sign", "light-on-dark"))
                else:
                    sign_key_variants.append(("sign", ds))
                sign_key_variants.extend([
                    ("import_data_sign", ds),
                    ("data_sign", ds),
                ])
            else:
                sign_key_variants = []

            job = None
            errors: List[Exception] = []
            for base_params in param_variants:
                params = {**base_params, **kwargs}
                # try sign variants
                sign_applied = False
                for k, v in sign_key_variants:
                    try_params = {**params, k: v}
                    try:
                        job = workspace.create_job("import_particles", params=try_params)
                        sign_applied = True
                        break
                    except Exception as e:
                        errors.append(e)
                        job = None
                if job is None and not sign_applied:
                    try:
                        job = workspace.create_job("import_particles", params=params)
                    except Exception as e:
                        errors.append(e)
                        job = None
                if job is not None:
                    break

            if job is None:
                raise RuntimeError(
                    "Unable to create import_particles job with provided STAR path; "
                    f"tried variants of params, last error: {errors[-1] if errors else 'unknown'}"
                )

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
            print(f"Queued import particles job: {job.uid}")

            self._job_cache[job.uid] = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "params": {"star_path": star_path, **kwargs},
            }

            result: Dict[str, Any] = {
                "job_uid": job.uid,
                "job_type": "import_particles",
                "status": "queued",
                "params": {"star_path": star_path, **kwargs},
                "lane": used_lane,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
            }

            if wait_for_completion:
                print(f"⏳ Waiting for import particles job {job.uid} to complete...")
                try:
                    final_status = self.wait_for_job_completion(
                        project_uid,
                        job.uid,
                        workspace_uid,
                        timeout,
                        check_interval,
                    )
                    result["status"] = final_status["status"]
                    result["final_status"] = final_status
                    if final_status["status"] == "completed":
                        print(f"✅ Import particles job {job.uid} completed successfully!")
                    else:
                        print(
                            f"⚠️ Import particles job {job.uid} finished with status: {final_status['status']}"
                        )
                except TimeoutError:
                    result["status"] = "timeout"
                    print(f"⏰ Import particles job {job.uid} timed out after {timeout} seconds")
                except Exception as e:
                    result["status"] = "error"
                    print(f"❌ Error monitoring import particles job {job.uid}: {e}")

            return result
        except Exception as e:
            raise RuntimeError(f"Failed to import particles from STAR: {e}")
    
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
    
    def blob_picker(
        self,
        project_uid: str,
        workspace_uid: str,
        micrographs_job_uid: str,
        particle_diameter: float,
        diameter_max: Optional[float] = None,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run blob picker to detect particles in micrographs.
        
        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            micrographs_job_uid: UID of the micrograph selection or CTF estimation job
            particle_diameter: Particle diameter in Angstroms (minimum diameter)
            diameter_max: Maximum particle diameter in Angstroms (defaults to 2 * diameter)
            lane: Compute lane to use
            hostname: Specific hostname to run on
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            check_interval: Time between status checks in seconds
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing job information
        """
        try:
            # Find project and workspace
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)
            
            # Set default diameter_max if not provided
            if diameter_max is None:
                diameter_max = particle_diameter * 2.0
            
            # Prepare job parameters for blob picker GPU
            # CryoSPARC expects: "diameter" (min) and "diameter_max" (max)
            job_params = {
                "diameter": particle_diameter,
                "diameter_max": diameter_max,
                **kwargs
            }
            
            # Create blob picker job with connections
            # Try different output labels from the curate_exposures_v2 job
            connection_errors = []
            job = None
            for output_label in ("exposures_accepted", "exposures", "selected_exposures"):
                try:
                    job = workspace.create_job(
                        "blob_picker_gpu",  # Correct job type!
                        params=job_params,
                        connections={"micrographs": (micrographs_job_uid, output_label)}  # Use "micrographs" key
                    )
                    print(f"✅ Connected blob picker to {micrographs_job_uid}.{output_label}")
                    break
                except Exception as exc:
                    connection_errors.append((output_label, exc))
                    job = None
            
            if job is None:
                error_messages = ", ".join(
                    f"output '{label}': {err}" for label, err in connection_errors
                ) or "unknown"
                raise RuntimeError(
                    f"Unable to connect blob picker to micrograph job outputs: {error_messages}"
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
            print(f"Queued blob picker GPU job: {job.uid}")
            
            self._job_cache[job.uid] = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            result = {
                "job_uid": job.uid,
                "job_type": "blob_picker_gpu",
                "status": "queued",
                "params": job_params,
                "connections": {"micrographs": micrographs_job_uid},
                "lane": used_lane,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            
            # Wait for completion if requested
            if wait_for_completion:
                print(f"⏳ Waiting for blob picker job {job.uid} to complete...")
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
                        print(f"✅ Blob picker job {job.uid} completed successfully!")
                    else:
                        print(f"⚠️ Blob picker job {job.uid} finished with status: {final_status['status']}")
                except TimeoutError:
                    result["status"] = "timeout"
                    print(f"⏰ Blob picker job {job.uid} timed out after {timeout} seconds")
                except Exception as e:
                    result["status"] = "error"
                    print(f"❌ Error monitoring blob picker job {job.uid}: {e}")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to start blob picker: {e}")
    
    def extract_particles(
        self,
        project_uid: str,
        workspace_uid: str,
        particles_job_uid: str,
        micrographs_job_uid: str,
        box_size_pix: int,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract particles from micrographs using particle coordinates.
        
        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            particles_job_uid: UID of the blob picker or picking job
            micrographs_job_uid: UID of the micrograph job (CTF or selection)
            box_size_pix: Box size for extraction in pixels
            lane: Compute lane to use
            hostname: Specific hostname to run on
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            check_interval: Time between status checks in seconds
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing job information
        """
        try:
            # Find project and workspace
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)
            
            # Prepare job parameters for particle extraction
            job_params = {
                "box_size_pix": box_size_pix,
                **kwargs
            }
            
            # Create extraction job with connections to both particles and micrographs
            # Try different output labels from the picker job and micrograph job
            connection_errors = []
            job = None
            
            # Try different combinations of output labels
            particle_labels = ("particles", "particles_all", "picked_particles")
            micrograph_labels = ("exposures_accepted", "micrographs", "exposures")
            
            for particle_label in particle_labels:
                for micrograph_label in micrograph_labels:
                    try:
                        job = workspace.create_job(
                            "extract_micrographs_multi",  # Particle extraction job type
                            params=job_params,
                            connections={
                                "particles": (particles_job_uid, particle_label),
                                "micrographs": (micrographs_job_uid, micrograph_label)
                            }
                        )
                        print(f"✅ Connected particle extraction:")
                        print(f"   - Particles: {particles_job_uid}.{particle_label}")
                        print(f"   - Micrographs: {micrographs_job_uid}.{micrograph_label}")
                        break
                    except Exception as exc:
                        connection_errors.append((particle_label, micrograph_label, exc))
                        job = None
                if job is not None:
                    break
            
            if job is None:
                error_messages = "\n".join(
                    f"  particles '{p_label}' + micrographs '{m_label}': {err}" 
                    for p_label, m_label, err in connection_errors
                ) or "unknown"
                raise RuntimeError(
                    f"Unable to connect particle extraction to job outputs:\n{error_messages}"
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
            print(f"Queued particle extraction job: {job.uid}")
            
            self._job_cache[job.uid] = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            result = {
                "job_uid": job.uid,
                "job_type": "extract_micrographs_multi",
                "status": "queued",
                "params": job_params,
                "connections": {
                    "particles": particles_job_uid,
                    "micrographs": micrographs_job_uid
                },
                "lane": used_lane,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            
            # Wait for completion if requested
            if wait_for_completion:
                print(f"⏳ Waiting for particle extraction job {job.uid} to complete...")
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
                        print(f"✅ Particle extraction job {job.uid} completed successfully!")
                    else:
                        print(f"⚠️ Particle extraction job {job.uid} finished with status: {final_status['status']}")
                except TimeoutError:
                    result["status"] = "timeout"
                    print(f"⏰ Particle extraction job {job.uid} timed out after {timeout} seconds")
                except Exception as e:
                    result["status"] = "error"
                    print(f"❌ Error monitoring particle extraction job {job.uid}: {e}")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to start particle extraction: {e}")
    
    def class_2d(
        self,
        project_uid: str,
        workspace_uid: str,
        particles_job_uid: str,
        num_classes: int = 20,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 7200,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform 2D classification on extracted particles.
        
        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            particles_job_uid: UID of the particle extraction job
            num_classes: Number of 2D classes to generate
            lane: Compute lane to use
            hostname: Specific hostname to run on
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            check_interval: Time between status checks in seconds
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing job information
        """
        try:
            # Find project and workspace
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)
            
            # Prepare job parameters for 2D classification
            job_params = {
                "class2D_K": num_classes,  # Number of classes
                **kwargs
            }
            
            # Create 2D classification job with connections
            # Try different output labels from the extraction job
            connection_errors = []
            job = None
            for output_label in ("particles", "particles_all", "extracted_particles"):
                try:
                    job = workspace.create_job(
                        "class_2D",  # 2D classification job type
                        params=job_params,
                        connections={"particles": (particles_job_uid, output_label)}
                    )
                    print(f"✅ Connected 2D classification to {particles_job_uid}.{output_label}")
                    break
                except Exception as exc:
                    connection_errors.append((output_label, exc))
                    job = None
            
            if job is None:
                error_messages = ", ".join(
                    f"output '{label}': {err}" for label, err in connection_errors
                ) or "unknown"
                raise RuntimeError(
                    f"Unable to connect 2D classification to extraction job outputs: {error_messages}"
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
            print(f"Queued 2D classification job: {job.uid}")
            
            self._job_cache[job.uid] = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            result = {
                "job_uid": job.uid,
                "job_type": "class_2D",
                "status": "queued",
                "params": job_params,
                "connections": {"particles": particles_job_uid},
                "lane": used_lane,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            
            # Wait for completion if requested
            if wait_for_completion:
                print(f"⏳ Waiting for 2D classification job {job.uid} to complete...")
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
                        print(f"✅ 2D classification job {job.uid} completed successfully!")
                    else:
                        print(f"⚠️ 2D classification job {job.uid} finished with status: {final_status['status']}")
                except TimeoutError:
                    result["status"] = "timeout"
                    print(f"⏰ 2D classification job {job.uid} timed out after {timeout} seconds")
                except Exception as e:
                    result["status"] = "error"
                    print(f"❌ Error monitoring 2D classification job {job.uid}: {e}")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to start 2D classification: {e}")
    
    def select_2d_classes(
        self,
        project_uid: str,
        workspace_uid: str,
        class_2d_job_uid: str,
        top_n_classes: int = 5,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 300,
        check_interval: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Select top N 2D classes based on particle count.
        
        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            class_2d_job_uid: UID of the 2D classification job
            top_n_classes: Number of top classes to select
            lane: Compute lane to use
            hostname: Specific hostname to run on
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            check_interval: Time between status checks in seconds
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing job information
        """
        try:
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)

            job_params: Dict[str, Any] = {}
            selection_metadata: Optional[Dict[str, Any]] = None

            # Create selection job with both particles and class averages connected
            job = workspace.create_job(
                "select_2D",
                params=job_params,
                connections={
                    "particles": (class_2d_job_uid, "particles"),
                    "templates": (class_2d_job_uid, "class_averages")
                }
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
            print(f"Queued 2D class selection job: {job.uid}")
            
            self._job_cache[job.uid] = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            result = {
                "job_uid": job.uid,
                "job_type": "select_2D",
                "status": "queued",
                "params": job_params,
                "connections": {
                    "particles": class_2d_job_uid,
                    "templates": class_2d_job_uid
                },
                "lane": used_lane,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }

            # Wait for interactive job to become ready (status 'waiting')
            selected_indices: List[int] = []
            try:
                job.wait_for_status("waiting", timeout=timeout)
            except Exception:
                # If the job transitions directly to running/completed we continue
                pass

            # Auto-select top N classes based on particle count
            try:
                class_info = job.interact("get_class_info")
                if isinstance(class_info, list) and class_info:
                    top_n = max(0, int(top_n_classes)) if top_n_classes is not None else 0
                    if top_n > 0:
                        sorted_classes = sorted(
                            class_info,
                            key=lambda c: c.get("num_particles_total", 0),
                            reverse=True
                        )
                        selected = sorted_classes[:min(top_n, len(sorted_classes))]
                        selected_indices = [int(entry["class_idx"]) for entry in selected]
                        for entry in class_info:
                            class_idx = int(entry.get("class_idx", -1))
                            should_select = class_idx in selected_indices
                            job.interact(
                                "set_class_selected",
                                {
                                    "class_idx": class_idx,
                                    "selected": should_select
                                }
                            )
                        selection_metadata = {
                            "requested_top_n": top_n,
                            "selected_indices": selected_indices,
                            "class_counts": {
                                int(entry.get("class_idx", -1)): int(entry.get("num_particles_total", 0))
                                for entry in selected
                            }
                        }
            except Exception as auto_select_error:
                print(f"⚠️ Unable to auto-select top classes interactively: {auto_select_error}")

            # Close the interactive session (continues even if finish fails)
            try:
                job.interact("finish")
            except Exception as finish_error:
                print(f"⚠️ select_2D finish interaction failed: {finish_error}")

            if selected_indices:
                result["selected_template_indices"] = selected_indices
            if selection_metadata:
                result["selection_metadata"] = selection_metadata

            # Wait for completion if requested
            if wait_for_completion:
                print(f"⏳ Waiting for 2D class selection job {job.uid} to complete...")
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
                        print(f"✅ 2D class selection job {job.uid} completed successfully!")
                    else:
                        print(f"⚠️ 2D class selection job {job.uid} finished with status: {final_status['status']}")
                except TimeoutError:
                    result["status"] = "timeout"
                    print(f"⏰ 2D class selection job {job.uid} timed out after {timeout} seconds")
                except Exception as e:
                    result["status"] = "error"
                    print(f"❌ Error monitoring 2D class selection job {job.uid}: {e}")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to start 2D class selection: {e}")
    
    def template_picker(
        self,
        project_uid: str,
        workspace_uid: str,
        micrographs_job_uid: str,
        template_job_uid: str,
        lowpass_resolution: float = 20.0,
        *,
        particle_diameter: Optional[float] = None,
        lowpass_micrograph: Optional[float] = None,
        angular_spacing_deg: Optional[float] = None,
        min_distance: Optional[float] = None,
        use_ctf: Optional[bool] = None,
        blob_picker_job_uid: Optional[str] = None,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Template-based particle picking using 2D class averages as templates.
        
        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            micrographs_job_uid: UID of the micrograph job (CTF or selection)
            template_job_uid: UID of the job containing template images (selected 2D classes)
            lowpass_resolution: Low-pass filter resolution in Angstroms
            lane: Compute lane to use
            hostname: Specific hostname to run on
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            check_interval: Time between status checks in seconds
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing job information
        """
        try:
            # Find project and workspace
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)
            
            job_params: Dict[str, Any] = {}

            if particle_diameter is None and blob_picker_job_uid:
                particle_diameter = self._infer_particle_diameter_from_blob(project_uid, blob_picker_job_uid)

            if particle_diameter is None:
                raise RuntimeError(
                    "Template picker requires particle_diameter but none was provided and it could not be inferred from blob picker job"
                )

            job_params["diameter"] = float(particle_diameter)

            if lowpass_resolution is not None:
                job_params["lowpass_res_template"] = float(lowpass_resolution)

            if lowpass_micrograph is not None:
                job_params["lowpass_res"] = float(lowpass_micrograph)
            elif lowpass_resolution is not None:
                job_params.setdefault("lowpass_res", float(lowpass_resolution))

            if angular_spacing_deg is not None:
                job_params["angular_spacing_deg"] = float(angular_spacing_deg)

            if min_distance is not None:
                job_params["min_distance"] = float(min_distance)

            if use_ctf is not None:
                job_params["use_ctf"] = bool(use_ctf)

            # Allow only supported additional parameters from kwargs
            allowed_optional_params = {
                "sigma_multiplier",
                "thresh_low",
                "thresh_high",
                "max_prune_dist",
                "rotation_step",
                "num_plot",
                "ice_multiplier"
            }
            for key, value in kwargs.items():
                if key in allowed_optional_params and value is not None:
                    job_params[key] = value
            
            # Create template picker job with connections to both micrographs and templates
            # Try different combinations of output labels
            connection_errors = []
            job = None
            
            # Try different combinations of output labels
            template_labels = (
                "templates_selected",
                "class_averages",
                "templates_excluded",
                "templates"
            )
            micrograph_labels = ("exposures_accepted", "micrographs", "exposures")
            
            for template_label in template_labels:
                for micrograph_label in micrograph_labels:
                    try:
                        job = workspace.create_job(
                            "template_picker_gpu",  # Template picker job type
                            params=job_params,
                            connections={
                                "templates": (template_job_uid, template_label),
                                "micrographs": (micrographs_job_uid, micrograph_label)
                            }
                        )
                        print(f"✅ Connected template picker:")
                        print(f"   - Templates: {template_job_uid}.{template_label}")
                        print(f"   - Micrographs: {micrographs_job_uid}.{micrograph_label}")
                        break
                    except Exception as exc:
                        connection_errors.append((template_label, micrograph_label, exc))
                        job = None
                if job is not None:
                    break
            
            if job is None:
                error_messages = "\n".join(
                    f"  templates '{t_label}' + micrographs '{m_label}': {err}" 
                    for t_label, m_label, err in connection_errors
                ) or "unknown"
                raise RuntimeError(
                    f"Unable to connect template picker to job outputs:\n{error_messages}"
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
            print(f"Queued template picker job: {job.uid}")
            
            self._job_cache[job.uid] = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            result = {
                "job_uid": job.uid,
                "job_type": "template_picker_gpu",
                "status": "queued",
                "params": job_params,
                "connections": {
                    "templates": template_job_uid,
                    "micrographs": micrographs_job_uid
                },
                "lane": used_lane,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            
            # Wait for completion if requested
            if wait_for_completion:
                print(f"⏳ Waiting for template picker job {job.uid} to complete...")
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
                        print(f"✅ Template picker job {job.uid} completed successfully!")
                    else:
                        print(f"⚠️ Template picker job {job.uid} finished with status: {final_status['status']}")
                except TimeoutError:
                    result["status"] = "timeout"
                    print(f"⏰ Template picker job {job.uid} timed out after {timeout} seconds")
                except Exception as e:
                    result["status"] = "error"
                    print(f"❌ Error monitoring template picker job {job.uid}: {e}")
            
            return result

        except Exception as e:
            raise RuntimeError(f"Failed to start template picker: {e}")

    def _infer_particle_diameter_from_blob(
        self,
        project_uid: str,
        blob_job_uid: str
    ) -> Optional[float]:
        cache_entry = self._job_cache.get(blob_job_uid, {})
        cached_params = cache_entry.get("params", {}) if isinstance(cache_entry, dict) else {}
        diameter = cached_params.get("diameter") or cached_params.get("particle_diameter")
        if diameter is not None:
            try:
                return float(diameter)
            except (TypeError, ValueError):
                pass

        try:
            job = self.cs.find_job(project_uid, blob_job_uid)
            doc = getattr(job, "doc", {})
            if isinstance(doc, dict):
                params_spec = doc.get("params_spec", {})
                if isinstance(params_spec, dict):
                    spec_entry = params_spec.get("diameter")
                    if isinstance(spec_entry, dict) and spec_entry.get("value") is not None:
                        return float(spec_entry["value"])
                params_base = doc.get("params_base", {})
                if isinstance(params_base, dict):
                    base_entry = params_base.get("diameter")
                    if isinstance(base_entry, dict) and base_entry.get("value") is not None:
                        return float(base_entry["value"])
        except Exception:
            pass

        if isinstance(diameter, (int, float)):
            return float(diameter)
        try:
            return float(diameter)  # handle numeric strings
        except (TypeError, ValueError):
            return None

    def get_job_output_directory(
        self,
        project_uid: str,
        job_uid: str
    ) -> Dict[str, Any]:
        """
        Get the output directory and related information for a job.
        
        Args:
            project_uid: CryoSPARC project UID
            job_uid: UID of the job
            
        Returns:
            Dictionary containing job directory information
        """
        try:
            job = self.cs.find_job(project_uid, job_uid)
            job.refresh()
            
            # Get job directory path and convert to string for JSON serialization
            job_dir = str(job.dir())
            
            # Get job document for additional information
            doc = getattr(job, "doc", {})
            
            # Try to get output information
            output_result_groups = doc.get("output_result_groups", [])
            output_info = []
            
            for group in output_result_groups:
                output_info.append({
                    "type": group.get("type"),
                    "name": group.get("name"),
                    "title": group.get("title"),
                    "num_items": group.get("num_items", 0)
                })
            
            return {
                "job_uid": job_uid,
                "project_uid": project_uid,
                "job_directory": job_dir,
                "job_type": doc.get("job_type"),
                "status": doc.get("status"),
                "outputs": output_info
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get job output directory for {job_uid}: {e}")
    
    def ab_initio_reconstruction(
        self,
        project_uid: str,
        workspace_uid: str,
        particles_job_uid: str,
        num_classes: int = 1,
        initial_resolution: float = 20.0,
        final_resolution: float = 10.0,
        max_iterations: int = 50,
        symmetry: str = "C1",
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run ab initio reconstruction to generate initial 3D model(s) from 2D particles.
        
        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            particles_job_uid: UID of the particles job (from 2D selection or extraction)
            num_classes: Number of 3D classes to generate (default: 1)
            initial_resolution: Starting resolution in Angstroms (default: 20.0)
            final_resolution: Target resolution in Angstroms (default: 10.0)
            max_iterations: Maximum number of iterations (default: 50)
            symmetry: Symmetry group (e.g., C1, C2, D7) (default: C1)
            lane: Compute lane to use
            hostname: Specific hostname to run on
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            check_interval: Time between status checks in seconds
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing job information
        """
        try:
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)
            
            # Create ab initio reconstruction job
            # Note: Start with empty params - CryoSPARC will use defaults
            # Parameter names may vary by CryoSPARC version
            job_params: Dict[str, Any] = {}
            
            # Try to add parameters only if we know they're valid
            # These names might not be correct - let CryoSPARC use defaults for now
            # TODO: Verify correct parameter names for your CryoSPARC version
            
            particles_output_slot = self._infer_particles_output_slot(project, particles_job_uid)
            
            job = workspace.create_job(
                "homo_abinit",  # Ab initio reconstruction job type
                connections={
                    "particles": (particles_job_uid, particles_output_slot)
                },
                params=job_params
            )
            
            # Queue the job with lane auto-detection
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
            print(f"Queued ab initio reconstruction job: {job.uid}")
            
            job_uid = job.uid
            
            result = {
                "success": True,
                "job_uid": job_uid,
                "job_type": "ab_initio_reconstruction",
                "message": f"Ab initio reconstruction job {job_uid} queued successfully",
                "num_classes": num_classes,
                "initial_resolution": initial_resolution,
                "final_resolution": final_resolution,
                "max_iterations": max_iterations,
                "symmetry": symmetry,
                "lane": used_lane
            }
            
            if wait_for_completion:
                status_result = self.wait_for_job_completion(
                    project_uid=project_uid,
                    job_uid=job_uid,
                    timeout=timeout,
                    check_interval=check_interval
                )
                result.update(status_result)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "job_type": "ab_initio_reconstruction",
                "message": f"Failed to queue ab initio reconstruction job: {str(e)}"
            }
    
    def homogeneous_refinement(
        self,
        project_uid: str,
        workspace_uid: str,
        particles_job_uid: str,
        volume_job_uid: str,
        refinement_resolution: Optional[float] = None,
        symmetry: str = "C1",
        # Advanced refinement parameters
        refine_do_init_scale_est: bool = True,
        refine_highpass_res: Optional[float] = None,
        refine_num_final_iterations: Optional[int] = None,
        refine_res_init: Optional[float] = None,
        refine_symmetry_do_align: bool = True,
        # Job control parameters
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run homogeneous refinement to refine a single 3D structure.
        
        Note: For homogeneous refinement, both particles and volume come from the ab initio job.
        The connections used are:
        - particles: (volume_job_uid, "particles_all_classes")
        - volume: (volume_job_uid, "volume_class_0")
        
        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            particles_job_uid: UID of particles job (kept for compatibility, not used in connections)
            volume_job_uid: UID of the ab initio job (used for both particles and volume)
            refinement_resolution: Target resolution in Angstroms (optional)
            symmetry: Symmetry group (e.g., C1, C2, D7) (default: C1)
            # Advanced refinement parameters
            refine_do_init_scale_est: Enable initial scale estimation (default: True)
            refine_highpass_res: High-pass filter resolution in Angstroms (optional)
            refine_num_final_iterations: Number of final refinement iterations (optional)
            refine_res_init: Initial resolution for refinement in Angstroms (optional)
            refine_symmetry_do_align: Enable symmetry alignment (default: True)
            # Job control parameters
            lane: Compute lane to use
            hostname: Specific hostname to run on
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            check_interval: Time between status checks in seconds
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing job information
        """
        try:
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)
            
            # Determine the correct output slots based on source job type
            try:
                source_job = project.find_job(volume_job_uid)
                source_job_type = source_job.doc.get("type", "")
                
                # Ab initio jobs (homo_abinit) use these slots:
                if "abinit" in source_job_type.lower():
                    particles_slot = "particles_all_classes"
                    volume_slot = "volume_class_0"
                # Homogeneous reconstruction (homo_recon) might use these:
                elif "recon" in source_job_type.lower():
                    # Try to detect actual output slots
                    particles_slot = "particles"  # Common for homo_recon
                    volume_slot = "volume"  # Common for homo_recon
                else:
                    # Default to ab initio convention
                    particles_slot = "particles_all_classes"
                    volume_slot = "volume_class_0"
                    
                print(f"ℹ️  Detected source job type: {source_job_type}")
                print(f"ℹ️  Using particles slot: '{particles_slot}', volume slot: '{volume_slot}'")
            except Exception as e:
                # Fallback to ab initio convention
                particles_slot = "particles_all_classes"
                volume_slot = "volume_class_0"
                print(f"⚠️  Could not detect job type, using default slots: {e}")
            
            # Create homogeneous refinement job with comprehensive parameters
            job_params: Dict[str, Any] = {
                "refine_do_init_scale_est": refine_do_init_scale_est,
                "refine_symmetry_do_align": refine_symmetry_do_align
            }
            
            # Add refinement resolution if specified
            if refinement_resolution is not None:
                job_params["refine_res"] = refinement_resolution
            
            # Add symmetry if specified and not C1
            if symmetry and symmetry != "C1":
                job_params["refine_symmetry"] = symmetry
            
            # Add high-pass filter resolution if specified
            if refine_highpass_res is not None:
                job_params["refine_highpass_res"] = refine_highpass_res
            
            # Add number of final iterations if specified
            if refine_num_final_iterations is not None:
                job_params["refine_num_final_iterations"] = refine_num_final_iterations
            
            # Add initial resolution if specified
            if refine_res_init is not None:
                job_params["refine_res_init"] = refine_res_init
            
            # Create the job - matches user's example exactly
            job = workspace.create_job(
                "homo_refine_new",  # Homogeneous refinement job type
                connections={
                    "particles": (volume_job_uid, particles_slot),
                    "volume": (volume_job_uid, volume_slot)
                },
                params=job_params
            )
            
            # Queue the job with lane auto-detection
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
            print(f"Queued homogeneous refinement job: {job.uid}")
            
            job_uid = job.uid
            
            result = {
                "success": True,
                "job_uid": job_uid,
                "job_type": "homogeneous_refinement",
                "message": f"Homogeneous refinement job {job_uid} queued successfully",
                "symmetry": symmetry,
                "refinement_resolution": refinement_resolution,
                "lane": used_lane
            }
            
            if wait_for_completion:
                status_result = self.wait_for_job_completion(
                    project_uid=project_uid,
                    job_uid=job_uid,
                    timeout=timeout,
                    check_interval=check_interval
                )
                result.update(status_result)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "job_type": "homogeneous_refinement",
                "message": f"Failed to queue homogeneous refinement job: {str(e)}"
            }
    
    def heterogeneous_refinement(
        self,
        project_uid: str,
        workspace_uid: str,
        particles_job_uid: str,
        volume_job_uids: List[str],
        num_classes: Optional[int] = None,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run heterogeneous refinement to refine multiple 3D structures simultaneously.
        
        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            particles_job_uid: UID of the particles job
            volume_job_uids: List of volume job UIDs (from ab initio)
            num_classes: Number of classes (default: length of volume_job_uids)
            lane: Compute lane to use
            hostname: Specific hostname to run on
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            check_interval: Time between status checks in seconds
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing job information
        """
        try:
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)
            
            # Determine number of classes
            if num_classes is None:
                num_classes = len(volume_job_uids)
            
            # Create heterogeneous refinement job
            # Note: Only set parameters that exist for hetrefine_new job type
            job_params: Dict[str, Any] = {
                "hetrefine_N": num_classes  # Number of classes
            }
            
            # Build connections for all volumes
            connections = {
                "particles": (particles_job_uid, "particles_selected" if "select" in particles_job_uid.lower() else "particles")
            }
            
            # Add volume connections
            for i, volume_job_uid in enumerate(volume_job_uids):
                connections[f"volume_{i}"] = (volume_job_uid, "volume")
            
            # Create the job
            job = workspace.create_job(
                "hetrefine_new",  # Heterogeneous refinement job type
                connections=connections,
                params=job_params
            )
            
            # Queue the job with lane auto-detection
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
            print(f"Queued heterogeneous refinement job: {job.uid}")
            
            job_uid = job.uid
            
            result = {
                "success": True,
                "job_uid": job_uid,
                "job_type": "heterogeneous_refinement",
                "message": f"Heterogeneous refinement job {job_uid} queued successfully",
                "num_classes": num_classes,
                "num_volumes": len(volume_job_uids),
                "lane": used_lane
            }
            
            if wait_for_completion:
                status_result = self.wait_for_job_completion(
                    project_uid=project_uid,
                    job_uid=job_uid,
                    timeout=timeout,
                    check_interval=check_interval
                )
                result.update(status_result)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "job_type": "heterogeneous_refinement",
                "message": f"Failed to queue heterogeneous refinement job: {str(e)}"
            }
    
    def get_job_log(self, job_uid: str, project_uid: Optional[str] = None, workspace_uid: Optional[str] = None) -> Dict[str, Any]:
        """
        Read the log file of a CryoSPARC job to analyze errors and failures.
        
        Args:
            job_uid: UID of the job to read logs for
            project_uid: Optional project UID containing the job
            workspace_uid: Optional workspace UID containing the job
            
        Returns:
            Dictionary containing log content and analysis
        """
        try:
            # Try to construct log file path using common CryoSPARC project structure
            # This is a more direct approach that doesn't rely on get_job_info
            log_file_paths = []
            
            # Try different possible log file locations
            if project_uid and workspace_uid:
                # Try the full path structure
                log_file_paths.append(f"/home/daoyi/cryosparc/cryosparc_projects/{project_uid}/{workspace_uid}/{job_uid}/job.log")
            
            # Try the example path structure from the user
            log_file_paths.append(f"/home/daoyi/cryosparc/cryosparc_projects/CS-test/{job_uid}/job.log")
            
            # Try to find the log file in any of the possible locations
            log_file_path = None
            for path in log_file_paths:
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        # Just try to read a small part to test if file exists
                        f.read(100)
                    log_file_path = path
                    break
                except (FileNotFoundError, PermissionError):
                    continue
            
            if not log_file_path:
                return {
                    "success": False,
                    "error": "Log file not found",
                    "message": f"Could not locate job log file for {job_uid} in any expected location"
                }
            
            # Read the full log content
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
                "job_uid": job_uid,
                "log_file_path": log_file_path,
                "log_content": log_content,
                "log_size": len(log_content),
                "error_analysis": error_analysis,
                "message": f"Successfully read log for job {job_uid}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to read job log for {job_uid}: {str(e)}"
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
            "symmetry_error": [
                "symmetry error", "invalid symmetry", "symmetry failed",
                "symmetry mismatch"
            ],
            "gpu_error": [
                "gpu error", "cuda error", "gpu memory", "gpu failed",
                "cuda out of memory", "gpu timeout"
            ],
            "timeout_error": [
                "timeout", "timed out", "time limit exceeded",
                "execution timeout"
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
            analysis["suggestions"].append("Consider reducing batch size or using fewer particles")
            analysis["suggestions"].append("Try reducing the resolution or using a smaller patch size")
        
        if "parameter_error" in analysis["error_types"]:
            analysis["suggestions"].append("Check parameter values and ranges")
            analysis["suggestions"].append("Verify input data format and compatibility")
        
        if "file_error" in analysis["error_types"]:
            analysis["suggestions"].append("Verify file paths and permissions")
            analysis["suggestions"].append("Check if input files exist and are accessible")
        
        if "convergence_error" in analysis["error_types"]:
            analysis["suggestions"].append("Try increasing max_iterations or adjusting convergence criteria")
            analysis["suggestions"].append("Consider using different initial parameters or symmetry")
        
        if "symmetry_error" in analysis["error_types"]:
            analysis["suggestions"].append("Try using C1 symmetry (no symmetry) as a starting point")
            analysis["suggestions"].append("Verify the symmetry parameter matches your expected structure")
        
        if "gpu_error" in analysis["error_types"]:
            analysis["suggestions"].append("Try running on CPU instead of GPU")
            analysis["suggestions"].append("Reduce GPU memory usage by decreasing batch size")
        
        if "timeout_error" in analysis["error_types"]:
            analysis["suggestions"].append("Increase timeout limits or reduce job complexity")
            analysis["suggestions"].append("Consider breaking the job into smaller parts")
        
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