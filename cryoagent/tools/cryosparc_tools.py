"""CryoSPARC tools for cryoEM image processing."""

import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from cryosparc.tools import CryoSPARC
from ..config.config_loader import CryoSPARCSettings
from .cryosift_tools import CryoSiftTools


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
        gainref_flip_x: Optional[bool] = None,
        gainref_flip_y: Optional[bool] = None,
        gainref_rotate_num: Optional[int] = None,
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
            gainref_flip_x: Whether to flip gain reference in X (CryoSPARC convention)
            gainref_flip_y: Whether to flip gain reference in Y (CryoSPARC convention)
            gainref_rotate_num: Number of 90° clockwise rotations to apply to gain reference
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
            }
            
            if gain_ref_path:
                job_params["gainref_path"] = gain_ref_path
            if gainref_flip_x is not None:
                job_params["gainref_flip_x"] = bool(gainref_flip_x)
            if gainref_flip_y is not None:
                job_params["gainref_flip_y"] = bool(gainref_flip_y)
            if gainref_rotate_num is not None:
                try:
                    job_params["gainref_rotate_num"] = int(gainref_rotate_num) % 4
                except (TypeError, ValueError):
                    print(f"Warning: Invalid gainref_rotate_num '{gainref_rotate_num}', skipping.")
            
            if kwargs:
                job_params.update(kwargs)
            
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
            
            # First, check which output labels are available without creating jobs
            # This prevents creating multiple jobs for failed connection attempts
            connection_errors = []
            valid_connection = None
            
            # Try different combinations of output labels
            # Include particles_selected for select_2D jobs
            particle_labels = ("particles", "particles_selected", "particles_all", "picked_particles", "selected_particles")
            micrograph_labels = ("exposures_accepted", "micrographs", "exposures")
            
            # Check which combination works by examining job outputs first
            try:
                particles_job = project.find_job(particles_job_uid)
                particles_job.refresh()
                particles_doc = getattr(particles_job, "doc", {})
                particles_outputs = particles_doc.get("output_result_groups", [])
                available_particle_labels = {group.get("name") for group in particles_outputs if group.get("type") == "particle"}
                
                micrographs_job = project.find_job(micrographs_job_uid)
                micrographs_job.refresh()
                micrographs_doc = getattr(micrographs_job, "doc", {})
                micrographs_outputs = micrographs_doc.get("output_result_groups", [])
                available_micrograph_labels = {group.get("name") for group in micrographs_outputs if "exposure" in group.get("type", "").lower() or "micrograph" in group.get("type", "").lower()}
                
                # Find first valid combination
                for particle_label in particle_labels:
                    if particle_label in available_particle_labels:
                        for micrograph_label in micrograph_labels:
                            if micrograph_label in available_micrograph_labels:
                                valid_connection = (particle_label, micrograph_label)
                                break
                        if valid_connection:
                            break
            except Exception as check_exc:
                # If checking fails, fall back to trial-and-error method
                pass
            
            # If we found a valid connection, use it; otherwise try all combinations
            if valid_connection:
                particle_label, micrograph_label = valid_connection
                try:
                    job = workspace.create_job(
                        "extract_micrographs_multi",
                        params=job_params,
                        connections={
                            "particles": (particles_job_uid, particle_label),
                            "micrographs": (micrographs_job_uid, micrograph_label)
                        }
                    )
                    print(f"✅ Connected particle extraction:")
                    print(f"   - Particles: {particles_job_uid}.{particle_label}")
                    print(f"   - Micrographs: {micrographs_job_uid}.{micrograph_label}")
                except Exception as exc:
                    connection_errors.append((particle_label, micrograph_label, exc))
                    job = None
            else:
                # Fallback: try all combinations (this may create multiple jobs)
                job = None
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
        particles_job_uid: Optional[str] = None,
        num_classes: int = 20,
        particles_group_name: Optional[str] = None,
        particles_job_uids: Optional[List[str]] = None,
        particles_group_names: Optional[List[str]] = None,
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
            particles_job_uid: UID of the particle extraction job (for single input, deprecated if particles_job_uids is provided)
            num_classes: Number of 2D classes to generate
            particles_group_name: Optional specific particles group name to use (e.g., "particles_excluded", "particles_selected")
            particles_job_uids: Optional list of particle job UIDs for multiple inputs (takes precedence over particles_job_uid)
            particles_group_names: Optional list of group names for each job in particles_job_uids
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
            
            # Support multiple particle inputs (for connecting both J159 and J157 when both functions are enabled)
            if particles_job_uids and len(particles_job_uids) > 1:
                # Multiple particle inputs: connect both jobs directly to class_2d
                connections = {}
                connection_errors = []
                
                # Infer group names if not provided
                if particles_group_names is None:
                    particles_group_names = []
                    for job_uid in particles_job_uids:
                        slot = self._infer_particles_output_slot(project, job_uid)
                        particles_group_names.append(slot)
                elif len(particles_group_names) != len(particles_job_uids):
                    # Pad or truncate to match length
                    inferred_names = []
                    for i, job_uid in enumerate(particles_job_uids):
                        if i < len(particles_group_names) and particles_group_names[i]:
                            inferred_names.append(particles_group_names[i])
                        else:
                            slot = self._infer_particles_output_slot(project, job_uid)
                            inferred_names.append(slot)
                    particles_group_names = inferred_names
                
                # Build connections dictionary with indexed connection names
                # CryoSPARC class_2d can accept multiple particle inputs using indexed names
                connection_key = []
                for i, (job_uid, group_name) in enumerate(zip(particles_job_uids, particles_group_names)):
                    # Try indexed connection names (particles_0, particles_1, etc.)
                    connection_key.append((job_uid, group_name))
                
                connections = {"particles": connection_key}
                
                # Try to create job with multiple connections
                try:
                    job = workspace.create_job(
                        "class_2D",  # 2D classification job type
                        params=job_params,
                        connections=connections
                    )
                    connected_jobs = ", ".join([f"{uid}.{name}" for uid, name in zip(particles_job_uids, particles_group_names)])
                    print(f"✅ Connected 2D classification to multiple particle jobs: {connected_jobs}")
                except Exception as exc:
                    # If indexed connections don't work, try alternative approach
                    # Some CryoSPARC versions might support multiple connections differently
                    connection_errors.append((f"multiple connections", exc))
                    raise RuntimeError(
                        f"Unable to connect 2D classification to multiple particle job inputs: {exc}. "
                        f"Jobs: {particles_job_uids}, Groups: {particles_group_names}"
                    )
            else:
                # Single particle input (backward compatible)
                # Use particles_job_uids[0] if provided, otherwise fall back to particles_job_uid
                single_job_uid = particles_job_uids[0] if particles_job_uids and len(particles_job_uids) == 1 else particles_job_uid
                
                if not single_job_uid:
                    raise ValueError("Either particles_job_uid or particles_job_uids must be provided")
                
                # Create 2D classification job with connections
                # If particles_group_name is specified, use it; otherwise try different output labels
                connection_errors = []
                job = None
                
                if particles_group_name:
                    # Use the specified group name
                    output_label = particles_group_name
                else:
                    # Try different output labels from the extraction job
                    output_label = "particles"
                
                
                try:
                    job = workspace.create_job(
                        "class_2D",  # 2D classification job type
                        params=job_params,
                        connections={"particles": (single_job_uid, output_label)}
                    )
                    print(f"✅ Connected 2D classification to {single_job_uid}.{output_label}")
                except Exception as exc:
                    connection_errors.append((output_label, exc))
                    job = None
                    raise RuntimeError(f"Unable to connect 2D classification to extraction job outputs: {exc}")
                    
                
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
            
            # Build connections info for result
            if particles_job_uids and len(particles_job_uids) > 1:
                # Multiple connections
                connections_info = {
                    f"particles_{i}" if i > 0 else "particles": job_uid 
                    for i, job_uid in enumerate(particles_job_uids)
                }
            else:
                # Single connection (backward compatible)
                single_job_uid = particles_job_uids[0] if particles_job_uids and len(particles_job_uids) == 1 else particles_job_uid
                connections_info = {"particles": single_job_uid}
            
            result = {
                "job_uid": job.uid,
                "job_type": "class_2D",
                "status": "queued",
                "params": job_params,
                "connections": connections_info,
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
        *,
        selection_mode: str = "top_n",
        cryosift_options: Optional[Dict[str, Any]] = None,
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
            top_n_classes: Number of top classes to select (used when selection_mode='top_n')
            selection_mode: Strategy for selecting classes ('top_n' or 'cryosift')
            cryosift_options: Additional arguments when selection_mode='cryosift'
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
            selection_metadata = {"strategy": selection_mode}
            try:
                job.wait_for_status("waiting", timeout=timeout)
            except Exception:
                # If the job transitions directly to running/completed we continue
                pass

            try:
                class_info = job.interact("get_class_info")
                if not isinstance(class_info, list) or not class_info:
                    class_info = []

                if selection_mode.lower() == "cryosift":
                    cryosift_cfg = cryosift_options or {}
                    try:
                        job_dir_info = self.get_job_output_directory(project_uid, class_2d_job_uid)
                        classification_dir = Path(job_dir_info.get("job_directory", "")).expanduser()
                        if not classification_dir.exists():
                            raise FileNotFoundError(
                                f"Classification directory not found: {classification_dir}"
                            )

                        output_dir = cryosift_cfg.get("output_dir")
                        if output_dir:
                            output_path = Path(output_dir).expanduser()
                        else:
                            subdir = cryosift_cfg.get("output_subdir", "cryosift_eval")
                            output_path = classification_dir / subdir

                        threshold = float(cryosift_cfg.get("threshold", 3.0))
                        weights_path = cryosift_cfg.get("weights_path")
                        python_executable = cryosift_cfg.get("python_executable", "python")
                        conda_env = cryosift_cfg.get("conda_env")
                        extra_args = cryosift_cfg.get("extra_args")

                        cryosift_tool = CryoSiftTools(
                            python_executable=python_executable,
                            conda_env=conda_env,
                        )

                        indices, scores, output_path = cryosift_tool.evaluate_and_get_selected_classes(
                            classification_dir,
                            output_path,
                            weights_path=weights_path,
                            threshold=threshold,
                            extra_args=extra_args,
                        )

                        selected_indices = [idx for idx in indices if idx is not None]
                        selection_metadata.update(
                            {
                                "threshold": threshold,
                                "scores": scores,
                                "output_directory": str(output_path),
                                "selection_mode": "cryosift",
                            }
                        )
                    except Exception as cryosift_error:
                        print(f"⚠️ CryoSift-based selection failed: {cryosift_error}")
                        fallback_strategy = cryosift_cfg.get("fallback_strategy", "top_n")
                        if fallback_strategy == "top_n":
                            selection_mode = "top_n"
                        else:
                            raise

                if selection_mode.lower() == "all":
                    # Select all classes
                    if class_info:
                        selected_indices = [int(entry.get("class_idx", -1)) for entry in class_info if entry.get("class_idx") is not None]
                        selection_metadata.update(
                            {
                                "total_classes": len(class_info),
                                "selected_classes": len(selected_indices),
                                "selection_mode": "all",
                            }
                        )
                        print(f"✅ Selected all {len(selected_indices)} classes via 'all' selection mode")
                    else:
                        selected_indices = []

                if selection_mode.lower() == "top_n":
                    top_n = max(0, int(top_n_classes)) if top_n_classes is not None else 0
                    if top_n > 0 and class_info:
                        sorted_classes = sorted(
                            class_info,
                            key=lambda c: c.get("num_particles_total", 0),
                            reverse=True,
                        )
                        selected = sorted_classes[: min(top_n, len(sorted_classes))]
                        selected_indices = [int(entry["class_idx"]) for entry in selected]
                        selection_metadata.update(
                            {
                                "requested_top_n": top_n,
                                "class_counts": {
                                    int(entry.get("class_idx", -1)): int(entry.get("num_particles_total", 0))
                                    for entry in selected
                                },
                                "selection_mode": "top_n",
                            }
                        )

                if selected_indices and selection_mode.lower() != "all":
                    # Print detailed list for non-"all" modes (avoid printing very long lists for "all" mode)
                    print(f"✅ Selected classes via {selection_metadata.get('selection_mode', selection_mode)}: {selected_indices}")

                if selected_indices and class_info:
                    valid_indices = {int(entry.get("class_idx", -1)) for entry in class_info}
                    filtered_indices = [idx for idx in selected_indices if idx in valid_indices]
                    selected_indices = filtered_indices
                    for entry in class_info:
                        class_idx = int(entry.get("class_idx", -1))
                        should_select = class_idx in selected_indices
                        job.interact(
                            "set_class_selected",
                            {
                                "class_idx": class_idx,
                                "selected": should_select,
                            },
                        )
            except Exception as auto_select_error:
                print(f"⚠️ Unable to auto-select classes: {auto_select_error}")

            # Close the interactive session (continues even if finish fails)
            try:
                job.interact("finish")
            except Exception as finish_error:
                print(f"⚠️ select_2D finish interaction failed: {finish_error}")

            if selected_indices:
                result["selected_template_indices"] = selected_indices
            if selection_metadata:
                selection_metadata["strategy"] = selection_metadata.get("selection_mode", selection_mode)
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
    
    def get_particle_count(
        self,
        project_uid: str,
        particles_job_uid: str,
        particles_group_name: str = "particles"
    ) -> Dict[str, Any]:
        """
        Get the number of particles in a particles job.
        
        Args:
            project_uid: CryoSPARC project UID
            particles_job_uid: UID of the particles job
            particles_group_name: Name of the particles output group (default: "particles")
            
        Returns:
            Dictionary with success, num_particles, and error if any
        """
        try:
            project = self.cs.find_project(project_uid)
            job = project.find_job(particles_job_uid)
            job.refresh()
            doc = getattr(job, "doc", {}) or {}
            outputs = doc.get("output_result_groups", []) or []
            
            # Find the particles group
            for group in outputs:
                if group.get("name") == particles_group_name:
                    num_items = group.get("num_items", 0)
                    return {
                        "success": True,
                        "num_particles": num_items,
                        "particles_group_name": particles_group_name,
                        "job_uid": particles_job_uid
                    }
            
            # If not found, try to infer
            particles_slot = self._infer_particles_output_slot(project, particles_job_uid)
            for group in outputs:
                if group.get("name") == particles_slot:
                    num_items = group.get("num_items", 0)
                    return {
                        "success": True,
                        "num_particles": num_items,
                        "particles_group_name": particles_slot,
                        "job_uid": particles_job_uid
                    }
            
            return {
                "success": False,
                "error": f"Could not find particles group '{particles_group_name}' or inferred slot in job {particles_job_uid}",
                "num_particles": 0
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "num_particles": 0
            }

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
    
    def get_refinement_fsc_info(
        self,
        project_uid: str,
        job_uid: str
    ) -> Dict[str, Any]:
        """
        Get FSC resolution and box size information from a refinement job output.
        
        Args:
            project_uid: CryoSPARC project UID
            job_uid: UID of the refinement job
            
        Returns:
            Dictionary containing:
            - box_size (N): Box size in pixels
            - resolution_angstroms (radwn_noisesub_A): FSC resolution in Angstroms
            - success: Whether the information was successfully retrieved
        """
        try:
            job = self.cs.find_job(project_uid, job_uid)
            job.refresh()
            doc = getattr(job, "doc", {})
            
            # Try to get latest summary stats from output_result_groups (plural, array)
            # Path: output_result_groups[i]->latest_summary_stats->fsc_info_best
            latest_summary_stats = None
            fsc_info = None
            
            if "output_result_groups" in doc:
                output_result_groups = doc.get("output_result_groups", [])
                if isinstance(output_result_groups, list):
                    # Try each output result group
                    for group in output_result_groups:
                        if isinstance(group, dict) and "latest_summary_stats" in group:
                            latest_summary_stats = group.get("latest_summary_stats", {})
                            fsc_info = latest_summary_stats.get("fsc_info_best")
                            if fsc_info:
                                break
            
            # Fallback: try output_result_group (singular)
            if not fsc_info and "output_result_group" in doc:
                output_result_group = doc.get("output_result_group", {})
                if isinstance(output_result_group, dict):
                    latest_summary_stats = output_result_group.get("latest_summary_stats", {})
                    fsc_info = latest_summary_stats.get("fsc_info_best") if latest_summary_stats else None
            
            # Fallback: try direct path in doc
            if not fsc_info:
                latest_summary_stats = doc.get("latest_summary_stats", {})
                fsc_info = latest_summary_stats.get("fsc_info_best") if latest_summary_stats else None
            
            # Final fallback: read from job.json file directly
            if not fsc_info:
                try:
                    job_dir = getattr(job, "dir", None)
                    if job_dir:
                        import json
                        from pathlib import Path
                        job_json_path = Path(job_dir) / "job.json"
                        if job_json_path.exists():
                            with open(job_json_path, 'r') as f:
                                file_data = json.load(f)
                            
                            # Search in output_result_groups
                            if "output_result_groups" in file_data:
                                output_result_groups = file_data.get("output_result_groups", [])
                                for group in output_result_groups:
                                    if isinstance(group, dict) and "latest_summary_stats" in group:
                                        latest_summary_stats = group.get("latest_summary_stats", {})
                                        fsc_info = latest_summary_stats.get("fsc_info_best")
                                        if fsc_info:
                                            break
                except Exception as file_error:
                    # File read failed, continue with error below
                    pass
            
            if not fsc_info:
                return {
                    "success": False,
                    "error": "No FSC information found in job output (checked output_result_groups, output_result_group, doc, and job.json file)"
                }
            
            # Extract box size (N) and resolution (radwn_noisesub_A)
            box_size = fsc_info.get("N")
            resolution_angstroms = fsc_info.get("radwn_noisesub_A")
            
            if box_size is None or resolution_angstroms is None:
                return {
                    "success": False,
                    "error": f"Missing FSC data: N={box_size}, radwn_noisesub_A={resolution_angstroms}"
                }
            
            return {
                "success": True,
                "box_size": int(box_size),
                "resolution_angstroms": float(resolution_angstroms),
                "fsc_info": fsc_info
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get FSC info from job {job_uid}: {str(e)}"
            }
    
    def get_heterogeneous_refinement_class_resolutions(
        self,
        project_uid: str,
        job_uid: str
    ) -> Dict[str, Any]:
        """
        Get resolution information for each class in a heterogeneous refinement job.
        
        For each class (volume_class_x where x is the class id starting from 0),
        extracts:
        - radwn_loosemask_A: Estimated resolution in Angstroms
        - fsc_loosemask: List of FSC values, use the last value for comparison
        
        Args:
            project_uid: CryoSPARC project UID
            job_uid: UID of the heterogeneous refinement job
            
        Returns:
            Dictionary containing:
            - success: Whether the information was successfully retrieved
            - classes: List of dictionaries, each with:
              - class_id: Class index (0, 1, 2, ...)
              - resolution_angstroms: Resolution in Angstroms (radwn_loosemask_A)
              - fsc_loosemask_last: Last value of fsc_loosemask list
              - group_name: Name of the output group (e.g., "volume_class_0")
            - error: Error message if unsuccessful
        """
        try:
            job = self.cs.find_job(project_uid, job_uid)
            job.refresh()
            doc = getattr(job, "doc", {})
            
            classes_info = []
            
            # Try to get from doc.output_result_groups
            output_result_groups = doc.get("output_result_groups", [])
            
            # Fallback: read from job.json file directly
            if not output_result_groups:
                try:
                    job_dir = getattr(job, "dir", None)
                    if job_dir:
                        import json
                        from pathlib import Path
                        job_json_path = Path(job_dir) / "job.json"
                        if job_json_path.exists():
                            with open(job_json_path, 'r') as f:
                                file_data = json.load(f)
                                output_result_groups = file_data.get("output_result_groups", [])
                except Exception as file_error:
                    pass
            
            if not output_result_groups:
                return {
                    "success": False,
                    "error": "No output_result_groups found in job output",
                    "classes": []
                }
            
            # Find all volume_class_x groups
            for group in output_result_groups:
                if not isinstance(group, dict):
                    continue
                
                group_name = group.get("name", "")
                
                # Check if this is a volume_class_x group
                if group_name.startswith("volume_class_"):
                    try:
                        # Extract class ID from name (e.g., "volume_class_0" -> 0)
                        class_id_str = group_name.replace("volume_class_", "")
                        class_id = int(class_id_str)
                        
                        # Get latest_summary_stats
                        latest_summary_stats = group.get("latest_summary_stats", {})
                        
                        # Extract resolution (radwn_loosemask_A)
                        resolution_angstroms = latest_summary_stats.get("radwn_loosemask_A")
                        
                        # Extract fsc_loosemask (should be a list)
                        fsc_loosemask = latest_summary_stats.get("fsc_loosemask", [])
                        fsc_loosemask_last = None
                        if isinstance(fsc_loosemask, list) and len(fsc_loosemask) > 0:
                            fsc_loosemask_last = float(fsc_loosemask[-1])
                        
                        if resolution_angstroms is not None:
                            classes_info.append({
                                "class_id": class_id,
                                "group_name": group_name,
                                "resolution_angstroms": float(resolution_angstroms),
                                "fsc_loosemask_last": fsc_loosemask_last,
                                "fsc_loosemask": fsc_loosemask
                            })
                    except (ValueError, TypeError, AttributeError) as e:
                        # Skip invalid groups
                        continue
            
            # Sort by class_id
            classes_info.sort(key=lambda x: x["class_id"])
            
            if not classes_info:
                return {
                    "success": False,
                    "error": "No volume_class_x groups found in output_result_groups",
                    "classes": []
                }
            
            return {
                "success": True,
                "classes": classes_info,
                "num_classes": len(classes_info)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get heterogeneous refinement class resolutions from job {job_uid}: {str(e)}",
                "classes": []
            }
    
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
            # Set parameters for homo_abinit job type
            # CryoSPARC parameter names for homo_abinit (from CryoSPARC v4.7.1):
            # - abinit_K: number of classes
            # - abinit_init_res: initial resolution in Angstroms (starting frequency)
            # - abinit_max_res: maximum resolution in Angstroms (maximum frequency)
            # - abinit_num_init_iters: number of initial iterations before annealing
            # - abinit_num_final_iters: number of final iterations after annealing
            # - abinit_symmetry: symmetry group
            job_params: Dict[str, Any] = {}
            
            # Set number of classes (K) - always set to ensure correct number of classes
            job_params["abinit_K"] = num_classes
            
            # Set resolution parameters (correct parameter names)
            job_params["abinit_init_res"] = initial_resolution
            job_params["abinit_max_res"] = final_resolution
            
            # Set iteration parameters
            # Note: CryoSPARC uses abinit_num_init_iters and abinit_num_final_iters
            # We'll set abinit_num_final_iters based on max_iterations
            # Default: abinit_num_init_iters=200, abinit_num_final_iters=300
            # Total iterations ≈ abinit_num_init_iters + annealing + abinit_num_final_iters
            # For simplicity, we'll set abinit_num_final_iters to approximate max_iterations
            if max_iterations is not None:
                # If max_iterations is provided, distribute it between init and final
                # Default init iters is 200, so we'll set final iters to max_iterations - 200
                # But ensure it's at least 100
                final_iters = max(100, max_iterations - 200)
                job_params["abinit_num_final_iters"] = final_iters
            
            # Set symmetry (only if not C1, otherwise CryoSPARC uses default)
            
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
            
            # Check if group names are explicitly provided (e.g., from heterogeneous refinement)
            particles_group_name = kwargs.get("particles_group_name")
            volume_group_name = kwargs.get("volume_group_name")
            
            if particles_group_name and volume_group_name:
                # Use explicitly provided group names (e.g., particles_class_X, volume_class_X)
                particles_slot = particles_group_name
                volume_slot = volume_group_name
                print(f"ℹ️  Using explicit group names: particles={particles_slot}, volume={volume_slot}")
            else:
                # Determine the correct output slots
                # For optimization: particles come from particles_job_uid (extraction), volume from volume_job_uid (reconstruction)
                # For normal refinement: both come from volume_job_uid (ab initio)
                
                # Check if particles_job_uid is different from volume_job_uid (optimization case)
                use_separate_particles = (particles_job_uid != volume_job_uid)
                
                if use_separate_particles:
                    # Optimization case: particles from extraction job, volume from refinement job
                    # Find particles output slot from extraction job
                    try:
                        extract_job = project.find_job(particles_job_uid)
                        extract_job.refresh()
                        extract_doc = getattr(extract_job, "doc", {})
                        extract_outputs = extract_doc.get("output_result_groups", [])
                        # Find particles output slot
                        particles_slot = "particles"  # Default
                        for group in extract_outputs:
                            if group.get("type") == "particle":
                                particles_slot = group.get("name", "particles")
                                break
                    except Exception as e:
                        particles_slot = "particles"
                        print(f"⚠️  Could not detect particles slot from extraction job, using default: {e}")
                    
                    # Find volume output slot from volume job (refinement job)
                    try:
                        volume_job = project.find_job(volume_job_uid)
                        volume_job.refresh()
                        volume_doc = getattr(volume_job, "doc", {})
                        volume_outputs = volume_doc.get("output_result_groups", [])
                        # Find volume output slot
                        volume_slot = "volume"  # Default
                        for group in volume_outputs:
                            if group.get("type") == "volume":
                                volume_slot = group.get("name", "volume")
                                break
                    except Exception as e:
                        volume_slot = "volume"
                        print(f"⚠️  Could not detect volume slot from volume job, using default: {e}")
                    
                    print(f"ℹ️  Optimization mode: particles from {particles_job_uid}.{particles_slot}, volume from {volume_job_uid}.{volume_slot}")
                else:
                    # Normal case: both from same job
                    try:
                        source_job = project.find_job(volume_job_uid)
                        source_job_type = source_job.doc.get("type", "")
                        
                        # Try to detect actual output slots from the job
                        source_job.refresh()
                        source_doc = getattr(source_job, "doc", {})
                        source_outputs = source_doc.get("output_result_groups", [])
                        
                        # Find actual particles and volume slots
                        detected_particles_slot = None
                        detected_volume_slot = None
                        
                        for group in source_outputs:
                            group_type = group.get("type", "").lower()
                            group_name = group.get("name", "")
                            if "particle" in group_type or "particle" in group_name.lower():
                                detected_particles_slot = group_name
                            elif "volume" in group_type or "volume" in group_name.lower():
                                detected_volume_slot = group_name
                        
                        # Use detected slots if found, otherwise fall back to job type conventions
                        if detected_particles_slot:
                            particles_slot = detected_particles_slot
                        elif "abinit" in source_job_type.lower():
                            particles_slot = "particles_all_classes"
                        elif "refine" in source_job_type.lower() or "recon" in source_job_type.lower():
                            particles_slot = "particles"  # Common for refinement/reconstruction jobs
                        else:
                            particles_slot = "particles"  # Default for other job types
                        
                        if detected_volume_slot:
                            volume_slot = detected_volume_slot
                        elif "abinit" in source_job_type.lower():
                            volume_slot = "volume_class_0"
                        elif "refine" in source_job_type.lower() or "recon" in source_job_type.lower():
                            volume_slot = "volume"  # Common for refinement/reconstruction jobs
                        else:
                            volume_slot = "volume"  # Default for other job types
                            
                        print(f"ℹ️  Detected source job type: {source_job_type}")
                        print(f"ℹ️  Using particles slot: '{particles_slot}', volume slot: '{volume_slot}'")
                    except Exception as e:
                        # Fallback: try to detect slots, otherwise use refinement convention
                        try:
                            source_job = project.find_job(volume_job_uid)
                            source_job.refresh()
                            source_doc = getattr(source_job, "doc", {})
                            source_outputs = source_doc.get("output_result_groups", [])
                            
                            for group in source_outputs:
                                group_type = group.get("type", "").lower()
                                group_name = group.get("name", "")
                                if "particle" in group_type:
                                    particles_slot = group_name
                                elif "volume" in group_type:
                                    volume_slot = group_name
                            print(f"ℹ️  Detected slots from job outputs: particles='{particles_slot}', volume='{volume_slot}'")
                        except Exception:
                            # Final fallback to refinement convention
                            particles_slot = "particles"
                            volume_slot = "volume"
                            print(f"⚠️  Could not detect job type, using refinement convention: {e}")
            
            # Create homogeneous refinement job with comprehensive parameters
            job_params: Dict[str, Any] = {
                "refine_do_init_scale_est": refine_do_init_scale_est,
                "refine_symmetry_do_align": refine_symmetry_do_align
            }
            
            # Add CTF refinement parameters if provided
            refine_defocus_refine = kwargs.get("refine_defocus_refine", True)
            refine_ctf_global_refine = kwargs.get("refine_ctf_global_refine", True)
            job_params["refine_defocus_refine"] = refine_defocus_refine
            job_params["refine_ctf_global_refine"] = refine_ctf_global_refine
            
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
                    "particles": (particles_job_uid, particles_slot),
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
    
    def reference_motion_correction(
        self,
        project_uid: str,
        workspace_uid: str,
        micrographs_job_uid: str,
        particles_job_uid: str,
        volume_job_uid: str,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run reference-based motion correction on particles using a reference volume.
        
        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            micrographs_job_uid: UID of the micrograph job (from preprocessing)
            particles_job_uid: UID of the particles job (from refinement)
            volume_job_uid: UID of the volume job (from refinement)
            lane: Compute lane to use
            hostname: Specific hostname to run on
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            check_interval: Time between status checks in seconds
            **kwargs: Additional parameters (all parameters from terminal selection can be passed here)
            
        Returns:
            Dictionary containing job information
        """
        try:
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)
            
            # Prepare job parameters - include all parameters from kwargs
            job_params: Dict[str, Any] = {}
            
            # Add all parameters from kwargs (these match the params_base from the terminal selection)
            # Common parameters that might be passed:
            allowed_params = [
                'max_processing_stage', 'num_reference_volumes', 'frame_start', 'frame_end',
                'bfactor', 'output_f16', 'recenter_particles', 'skip_align', 'skip_mismatching_frames',
                'align_cutoff_frac_hyp', 'hyparam_search_thoroughness', 'hypopt_minpcls', 'hypopt_rmax',
                'override_h1', 'override_h2', 'override_h3', 'use_all_fcs_dosewt', 'dosewt_minpcls',
                'use_all_fcs_opttraj', 'fcrop_box_size', 'output_fcrop_factor', 'compute_num_gpus',
                'gpu_oversub_gb', 'mem_cache_sz', 'slicing_gpu_is_worker', 'random_seed', 'eer_numfractions'
            ]
            
            for param in allowed_params:
                if param in kwargs:
                    job_params[param] = kwargs[param]
            
            # Add any other parameters from kwargs
            for key, value in kwargs.items():
                if key not in ['project_uid', 'workspace_uid', 'micrographs_job_uid', 
                              'particles_job_uid', 'volume_job_uid', 'lane', 'hostname',
                              'wait_for_completion', 'timeout', 'check_interval']:
                    if key not in job_params:
                        job_params[key] = value
            
            # Determine output slots
            # Micrographs: typically "exposures_accepted" or "micrograph"
            # Particles: typically "particles" or "particles_0"
            # Volume: typically "volume" or "volume_0"
            
            micrograph_slot = "exposures_accepted"  # Default
            particles_slot = "particles"  # Default
            volume_slot = "volume"  # Default
            
            # Try to detect actual slots
            try:
                micrograph_job = project.find_job(micrographs_job_uid)
                micrograph_job.refresh()
                micrograph_doc = getattr(micrograph_job, "doc", {})
                micrograph_outputs = micrograph_doc.get("output_result_groups", [])
                for group in micrograph_outputs:
                    if group.get("type") == "exposure":
                        micrograph_slot = group.get("name", "exposures_accepted")
                        break
            except Exception:
                pass
            
            try:
                particles_job = project.find_job(particles_job_uid)
                particles_job.refresh()
                particles_doc = getattr(particles_job, "doc", {})
                particles_outputs = particles_doc.get("output_result_groups", [])
                for group in particles_outputs:
                    if group.get("type") == "particle":
                        particles_slot = group.get("name", "particles")
                        break
            except Exception:
                pass
            
            try:
                volume_job = project.find_job(volume_job_uid)
                volume_job.refresh()
                volume_doc = getattr(volume_job, "doc", {})
                volume_outputs = volume_doc.get("output_result_groups", [])
                for group in volume_outputs:
                    if group.get("type") == "volume":
                        volume_slot = group.get("name", "volume")
                        break
            except Exception:
                pass
            
            # Create reference motion correction job
            job = workspace.create_job(
                "reference_motion_correction",
                params=job_params,
                connections={
                    "micrograph": (micrographs_job_uid, micrograph_slot),
                    "particles_0": (particles_job_uid, particles_slot),
                    "volume_0": (volume_job_uid, volume_slot)
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
            
            print(f"Queued reference motion correction job: {job.uid}")
            
            self._job_cache[job.uid] = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            
            result = {
                "success": True,
                "job_uid": job.uid,
                "job_type": "reference_motion_correction",
                "message": f"Reference motion correction job {job.uid} queued successfully",
                "lane": used_lane
            }
            
            if wait_for_completion:
                status_result = self.wait_for_job_completion(
                    project_uid=project_uid,
                    job_uid=job.uid,
                    workspace_uid=workspace_uid,
                    timeout=timeout,
                    check_interval=check_interval
                )
                result.update(status_result)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "job_type": "reference_motion_correction",
                "message": f"Failed to queue reference motion correction job: {str(e)}"
            }
    
    def heterogeneous_refinement(
        self,
        project_uid: str,
        workspace_uid: str,
        particles_job_uid: str,
        volume_job_uids: List[str],
        num_classes: Optional[int] = None,
        symmetry: str = "C1",
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
            symmetry: Symmetry group (e.g., C1, D7) - applied to all classes (default: C1)
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
            
            # Determine number of classes (will be inferred from volume connections)
            if num_classes is None:
                num_classes = len(volume_job_uids)
            
            # Create heterogeneous refinement job
            # Note: For hetero_refine job type, the number of classes is automatically determined
            # from the number of volume connections
            # The symmetry parameter is "multirefine_symmetry" for hetero_refine
            job_params: Dict[str, Any] = {}
            
            # Add symmetry if specified and not C1
            if symmetry and symmetry != "C1":
                job_params["multirefine_symmetry"] = symmetry
            
            # Build connections for all volumes
            # For hetero_refine, the input group is named "volume" (singular) and accepts multiple connections
            # The number of classes is determined by the number of volume connections to this single "volume" group
            # For K classes, we need to connect the SAME volume K times (repeat the same volume connection)
            particles_slot = "particles_selected" if "select" in particles_job_uid.lower() else "particles"
            
            # Determine the correct volume output slot from the first volume job
            # (all volumes should use the same slot since they're the same job repeated)
            volume_slot = "volume"  # Default
            try:
                first_volume_job = project.find_job(volume_job_uids[0])
                first_volume_job.refresh()
                volume_doc = getattr(first_volume_job, "doc", {})
                volume_outputs = volume_doc.get("output_result_groups", [])
                for group in volume_outputs:
                    if group.get("type") == "volume":
                        volume_slot = group.get("name", "volume")
                        break
            except Exception as e:
                volume_slot = "volume"
                print(f"⚠️  Could not detect volume slot, using default 'volume': {e}")
            
            connections = {
                "particles": (particles_job_uid, particles_slot)
            }
            
            # Add all volume connections to the single "volume" input group
            # For K classes, repeat the same (volume_job_uid, volume_slot) tuple K times
            # CryoSPARC will infer num_classes from the number of connections to "volume"
            # The Python API expects a list of tuples when an input group accepts multiple connections
            volume_connections = [(volume_job_uid, volume_slot) for volume_job_uid in volume_job_uids]
            connections["volume"] = volume_connections
            
            print(f"ℹ️  Heterogeneous refinement: connecting {len(volume_connections)} volumes (K={len(volume_connections)}) to 'volume' input group")
            print(f"ℹ️  All volumes from: {volume_job_uids[0]} (same volume repeated {len(volume_connections)} times)")
            
            # Create the job
            # The number of classes (K) is automatically determined from the number of volume connections
            job = workspace.create_job(
                "hetero_refine",  # Heterogeneous refinement job type
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
    
    def regroup_classes(
        self,
        project_uid: str,
        workspace_uid: str,
        particles_job_uid: str,
        num_superclasses: int = 2,
        job_title: Optional[str] = None,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Regroup K classes from a heterogeneous refinement into fewer superclasses.
        
        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            particles_job_uid: UID of the heterogeneous refinement job (contains particles_class_X groups)
            num_superclasses: Number of superclasses to create (default: 2)
            job_title: Optional title for the regroup job
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
            
            # Get the heterogeneous refinement job to find the combined output groups
            hetero_job = project.find_job(particles_job_uid)
            hetero_job.refresh()
            hetero_doc = getattr(hetero_job, "doc", {})
            hetero_outputs = hetero_doc.get("output_result_groups", [])
            
            # Count the number of input classes from the heterogeneous refinement job
            # Look for particles_class_X groups to determine K
            num_input_classes = 0
            for group in hetero_outputs:
                group_name = group.get("name", "")
                if group_name.startswith("particles_class_") and group_name != "particles_all_classes":
                    num_input_classes += 1
            
            # Special case: if we have exactly 2 input classes, skip regroup and select the best class
            # This will be used for direct homogeneous_refinement on the selected class
            if num_input_classes == 2:
                print(f"ℹ️  Regroup: K=2, selecting best class instead of creating regroup job")
                
                # Get resolution information for all classes
                resolutions_result = self.get_heterogeneous_refinement_class_resolutions(
                    project_uid=project_uid,
                    job_uid=particles_job_uid
                )
                
                if not resolutions_result.get("success"):
                    return {
                        "success": False,
                        "error": resolutions_result.get("error", "Failed to get class resolutions"),
                        "message": "Could not get class resolutions to select best class"
                    }
                
                classes = resolutions_result.get("classes", [])
                if not classes:
                    return {
                        "success": False,
                        "error": "No classes found in heterogeneous refinement job",
                        "message": "Cannot select best class: no classes available"
                    }
                
                # Find the best class: lower resolution is better
                # If resolution is the same (within tolerance), higher fsc_loosemask_last is better
                best_class = None
                best_resolution = float('inf')
                best_fsc_last = -1.0
                resolution_tolerance = 0.001  # 0.001 Å tolerance for resolution comparison
                
                for class_info in classes:
                    resolution = class_info.get("resolution_angstroms")
                    fsc_last = class_info.get("fsc_loosemask_last")
                    
                    if resolution is None:
                        continue
                    
                    # Lower resolution is better
                    if resolution < best_resolution - resolution_tolerance:
                        best_class = class_info
                        best_resolution = resolution
                        best_fsc_last = fsc_last if fsc_last is not None else -1.0
                    elif abs(resolution - best_resolution) <= resolution_tolerance:
                        # If resolution is the same (within tolerance), prefer higher fsc_loosemask_last
                        if fsc_last is not None and fsc_last > best_fsc_last:
                            best_class = class_info
                            best_resolution = resolution  # Update to current resolution
                            best_fsc_last = fsc_last
                
                if best_class is None:
                    return {
                        "success": False,
                        "error": "Could not determine best class",
                        "message": "No valid class with resolution data found"
                    }
                
                best_class_id = best_class.get("class_id")
                # The group_name from get_heterogeneous_refinement_class_resolutions is volume_class_X
                # We need to convert it to particles_class_X for the particles group
                volume_group_name = best_class.get("group_name")  # e.g., "volume_class_0"
                best_particles_group_name = volume_group_name.replace("volume_class_", "particles_class_")  # e.g., "particles_class_0"
                
                # Find the corresponding volume group name
                best_volume_group_name = volume_group_name  # Use the volume group name directly
                
                if not best_volume_group_name:
                    return {
                        "success": False,
                        "error": f"Could not find volume group for class {best_class_id}",
                        "message": f"Volume group 'volume_class_{best_class_id}' not found in heterogeneous refinement output"
                    }
                
                print(f"✅ Selected best class: {best_particles_group_name} (class {best_class_id})")
                print(f"   Resolution: {best_resolution:.3f} Å")
                if best_fsc_last is not None and best_fsc_last >= 0:
                    print(f"   FSC loosemask last value: {best_fsc_last:.4f}")
                print(f"   Volume group: {best_volume_group_name}")
                
                return {
                    "success": True,
                    "job_uid": None,  # No job created
                    "job_type": "class_selection",
                    "message": f"Selected best class {best_class_id} ({best_particles_group_name}) instead of regroup",
                    "num_superclasses": 1,
                    "selected_class": {
                        "class_id": best_class_id,
                        "particles_group_name": best_particles_group_name,
                        "volume_group_name": best_volume_group_name,
                        "resolution_angstroms": best_resolution,
                        "fsc_loosemask_last": best_fsc_last
                    }
                }
            
            # Normal regroup flow: create regroup job to regroup K classes into num_superclasses
            # Based on inspection of J202, regroup_3D_new connects to:
            # 1. "particles_all_classes" group (not individual particles_class_X groups)
            # 2. "volumes_all_classes" group (optional but recommended)
            
            # Find the combined output groups
            particles_all_classes = None
            volumes_all_classes = None
            
            for group in hetero_outputs:
                group_name = group.get("name", "")
                if group_name == "particles_all_classes":
                    particles_all_classes = group_name
                elif group_name == "volumes_all_classes":
                    volumes_all_classes = group_name
            
            if not particles_all_classes:
                return {
                    "success": False,
                    "error": f"No 'particles_all_classes' group found in job {particles_job_uid}",
                    "message": "Regroup requires heterogeneous refinement output with 'particles_all_classes' group. "
                              "This group contains all particle classes combined."
                }
            
            # Build connections for regroup job
            # The regroup job expects:
            # - "particles" input slot -> connects to "particles_all_classes" from hetero job
            # - "volume_series" input slot -> connects to "volumes_all_classes" from hetero job (optional)
            connections = {
                "particles": (particles_job_uid, particles_all_classes)
            }
            
            # Add volumes connection if available
            if volumes_all_classes:
                connections["volume_series"] = (particles_job_uid, volumes_all_classes)
            
            print(f"ℹ️  Regroup: connecting to '{particles_all_classes}' from job {particles_job_uid}")
            if volumes_all_classes:
                print(f"ℹ️  Regroup: also connecting to '{volumes_all_classes}' from job {particles_job_uid}")
            print(f"ℹ️  Regroup: will create {num_superclasses} superclasses")
            
            # Check available job types to find the correct regroup job type
            # The job type is "regroup_3D_new" not "regroup"
            regroup_job_type = "regroup_3D_new"  # Default to the correct job type name
            
            # Create the job with the number of superclasses parameter
            # The parameter name is 'regroup3D_N_K'
            job_params = {
                "regroup3D_N_K": num_superclasses
            }
            
            try:
                job = workspace.create_job(
                    regroup_job_type,
                    connections=connections,
                    params=job_params
                )
                print(f"✅ Created regroup job with regroup3D_N_K={num_superclasses}")
            except Exception as e:
                raise RuntimeError(f"Failed to create regroup job: {e}") from e
            
            # Set job title if provided (after job creation)
            if job_title:
                try:
                    job.set_title(job_title)
                except Exception as title_error:
                    # If set_title doesn't work, try alternative methods
                    print(f"⚠️  Could not set job title: {title_error}")
            
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
            print(f"Queued regroup job: {job.uid}")
            
            job_uid = job.uid
            
            result = {
                "success": True,
                "job_uid": job_uid,
                "job_type": "regroup_3D_new",
                "message": f"Regroup job {job_uid} queued successfully",
                "num_superclasses": num_superclasses,
                "lane": used_lane
            }
            
            if wait_for_completion:
                status_result = self.wait_for_job_completion(
                    project_uid=project_uid,
                    job_uid=job_uid,
                    workspace_uid=workspace_uid,
                    timeout=timeout,
                    check_interval=check_interval
                )
                result.update(status_result)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "job_type": "regroup_3D_new",
                "message": f"Failed to queue regroup job: {str(e)}"
            }
    
    def get_regroup_superclass_info(
        self,
        project_uid: str,
        job_uid: str
    ) -> Dict[str, Any]:
        """
        Get information about superclasses from a regroup job.
        Reads job.json to find num_items for each particles_superclass_x group.
        
        Args:
            project_uid: CryoSPARC project UID
            job_uid: UID of the regroup job
            
        Returns:
            Dictionary containing:
            - success: Whether the information was successfully retrieved
            - superclasses: List of dictionaries, each with:
              - superclass_id: Superclass index (0, 1, ...)
              - num_items: Number of particles in this superclass
              - group_name: Name of the output group (e.g., "particles_superclass_0")
            - error: Error message if unsuccessful
        """
        try:
            job = self.cs.find_job(project_uid, job_uid)
            job.refresh()
            doc = getattr(job, "doc", {})
            
            superclasses_info = []
            
            # Try to get from doc.output_result_groups
            output_result_groups = doc.get("output_result_groups", [])
            
            # Fallback: read from job.json file directly
            if not output_result_groups:
                try:
                    job_dir = getattr(job, "dir", None)
                    if job_dir:
                        import json
                        from pathlib import Path
                        job_json_path = Path(job_dir) / "job.json"
                        if job_json_path.exists():
                            with open(job_json_path, 'r') as f:
                                file_data = json.load(f)
                                output_result_groups = file_data.get("output_result_groups", [])
                except Exception as file_error:
                    pass
            
            if not output_result_groups:
                return {
                    "success": False,
                    "error": "No output_result_groups found in regroup job output",
                    "superclasses": []
                }
            
            # Find all particles_superclass_x groups
            for group in output_result_groups:
                group_name = group.get("name", "")
                if group_name.startswith("particles_superclass_"):
                    # Extract superclass ID from name (e.g., "particles_superclass_0" -> 0)
                    try:
                        superclass_id = int(group_name.split("_")[-1])
                        num_items = group.get("num_items", 0)
                        
                        superclasses_info.append({
                            "superclass_id": superclass_id,
                            "num_items": num_items,
                            "group_name": group_name
                        })
                    except (ValueError, IndexError):
                        # Skip if we can't parse the superclass ID
                        continue
            
            if not superclasses_info:
                return {
                    "success": False,
                    "error": "No particles_superclass_x groups found in regroup job output",
                    "superclasses": []
                }
            
            # Sort by superclass_id
            superclasses_info.sort(key=lambda x: x["superclass_id"])
            
            return {
                "success": True,
                "job_uid": job_uid,
                "num_superclasses": len(superclasses_info),
                "superclasses": superclasses_info
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "superclasses": []
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
    
    def import_volumes(
        self,
        project_uid: str,
        workspace_uid: str,
        half_map_a_path: str,
        half_map_b_path: str,
        pixel_size: Optional[float] = None,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Import two half maps (volumes) into CryoSPARC for FSC validation.
        
        This function creates two separate import_volumes jobs, one for each half map,
        with appropriate volume_out_name settings (map_half_A and map_half_B).
        
        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            half_map_a_path: Path to half map A (e.g., run_half1_class001_unfil.mrc)
            half_map_b_path: Path to half map B (e.g., run_half2_class001_unfil.mrc)
            pixel_size: Pixel size in Angstroms (optional, will try to infer from volumes)
            lane: Compute lane to use
            hostname: Specific hostname to run on
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            check_interval: Time between status checks in seconds
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing job information with imported volume job UIDs for both half maps
        """
        try:
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)
            
            # Import half map A
            job_params_a = {
                "volume_blob_path": half_map_a_path,
                "volume_out_name": "map_half_A"  # Specify this is half map A
            }
            
            if pixel_size is not None:
                job_params_a["volume_psize"] = float(pixel_size)
            
            if kwargs:
                job_params_a.update(kwargs)
            
            print(f"📤 Creating import job for half map A: {half_map_a_path}")
            job_a = workspace.create_job("import_volumes", params=job_params_a)
            
            # Import half map B
            job_params_b = {
                "volume_blob_path": half_map_b_path,
                "volume_out_name": "map_half_B"  # Specify this is half map B
            }
            
            if pixel_size is not None:
                job_params_b["volume_psize"] = float(pixel_size)
            
            if kwargs:
                job_params_b.update(kwargs)
            
            print(f"📤 Creating import job for half map B: {half_map_b_path}")
            job_b = workspace.create_job("import_volumes", params=job_params_b)
            
            # Queue both jobs
            used_lane = lane
            try:
                job_a.queue(lane=lane, hostname=hostname)
                job_b.queue(lane=lane, hostname=hostname)
            except Exception as queue_error:
                message = str(queue_error)
                if (lane is None and hostname is None and "Must specify a lane" in message):
                    try:
                        lanes = self.cs.get_lanes()
                        if not lanes:
                            raise queue_error
                        used_lane = lanes[0]["name"]
                        print(f"⚙️ No lane specified; retrying queue on lane '{used_lane}'")
                        job_a.queue(lane=used_lane)
                        job_b.queue(lane=used_lane)
                    except Exception:
                        raise queue_error
                else:
                    raise queue_error
            
            print(f"Queued import volumes job A: {job_a.uid}")
            print(f"Queued import volumes job B: {job_b.uid}")
            
            self._job_cache[job_a.uid] = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            self._job_cache[job_b.uid] = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            
            result = {
                "success": True,
                "job_uid_a": job_a.uid,
                "job_uid_b": job_b.uid,
                "job_type": "import_volumes",
                "status": "queued",
                "half_map_a_path": half_map_a_path,
                "half_map_b_path": half_map_b_path,
                "params": {"job_a": job_params_a, "job_b": job_params_b},
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "lane": used_lane
            }
            
            if wait_for_completion:
                print(f"⏳ Waiting for import volumes jobs to complete...")
                try:
                    final_status_a = self.wait_for_job_completion(
                        project_uid=project_uid,
                        job_uid=job_a.uid,
                        workspace_uid=workspace_uid,
                        timeout=timeout,
                        check_interval=check_interval
                    )
                    final_status_b = self.wait_for_job_completion(
                        project_uid=project_uid,
                        job_uid=job_b.uid,
                        workspace_uid=workspace_uid,
                        timeout=timeout,
                        check_interval=check_interval
                    )
                    
                    result["status_a"] = final_status_a["status"]
                    result["status_b"] = final_status_b["status"]
                    result["final_status_a"] = final_status_a
                    result["final_status_b"] = final_status_b
                    
                    if final_status_a["status"] == "completed" and final_status_b["status"] == "completed":
                        print(f"✅ Both import volumes jobs completed successfully!")
                    else:
                        print(f"⚠️ Import volumes jobs finished with status: A={final_status_a['status']}, B={final_status_b['status']}")
                except TimeoutError:
                    result["status"] = "timeout"
                    print(f"⏰ Import volumes jobs timed out after {timeout} seconds")
                except Exception as e:
                    result["status"] = "error"
                    print(f"❌ Error monitoring import volumes jobs: {e}")
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "job_type": "import_volumes",
                "message": f"Failed to import volumes: {str(e)}"
            }
    
    def compute_fsc_validation(
        self,
        project_uid: str,
        workspace_uid: str,
        volume_a_job_uid: str,
        volume_b_job_uid: str,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute FSC (Fourier Shell Correlation) between two half maps using CryoSPARC validation tools.

        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            volume_a_job_uid: UID of the job containing half map A
            volume_b_job_uid: UID of the job containing half map B
            lane: Compute lane to use
            hostname: Specific hostname to run on
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            check_interval: Time between status checks in seconds
            **kwargs: Additional parameters

        Returns:
            Dictionary containing job information and FSC results
        """
        try:
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)

            # 1. Prepare job parameters for FSC validation
            job_params: Dict[str, Any] = {
                "validate_generate_new_mask": True,  # Generate new FSC mask
                "validate_optimize_fsc_mask": True   # Optimize FSC mask
            }

            excluded_params = {'volume_a_slot', 'volume_b_slot'}
            valid_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_params}

            if valid_kwargs:
                job_params.update(valid_kwargs)

            # 2. Create the Validation Job (WITHOUT connections initially)
            # We use 'validation_fsc' which is the standard internal type for this job
            job_type = "validation"

            try:
                job = workspace.create_job(
                    type=job_type,
                    params=job_params
                    # Note: We do NOT pass 'connections' here. We handle them manually below.
                )
                print(f"✅ Created validation job using type '{job_type}' with UID {job.uid}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create validation job with type '{job_type}'. "
                    f"Error: {e}"
                )

            # 3. Connect using the High-Level API (Corrected)
            # We use job.connect() with the 'slots' parameter to handle the aliasing.
            # This handles the server communication format automatically.

            try:
                # Connect Half Map A
                # We map the source 'map' -> destination 'map_half_A'
                job.connect(
                    target_input="volume",           # The input group on the Validation job
                    source_job_uid=volume_a_job_uid,
                    source_output="imported_volume_1", # The output group from the Import job
                )

                # Connect Half Map B
                # We map the source 'map' -> destination 'map_half_B'
                # Calling connect() a second time on the same input appends this connection
                job.connect_result(
                    target_input="volume",
                    connection_idx = 0,
                    slot = "map_half_B",
                    source_job_uid=volume_b_job_uid,
                    source_output="imported_volume_1",
                    source_result = "map_half_B"
                )

                print(f"✅ Connected half-maps to validation job {job.uid}")

            except Exception as conn_err:
                raise RuntimeError(f"Failed to connect inputs: {conn_err}")

            # 4. Queue the job with lane auto-detection
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

            print(f"Queued FSC validation job: {job.uid}")

            # 5. Cache and Setup Result Object
            self._job_cache[job.uid] = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }

            result = {
                "success": True,
                "job_uid": job.uid,
                "job_type": "compute_fsc_validation",
                "status": "queued",
                "volume_a_job_uid": volume_a_job_uid,
                "volume_b_job_uid": volume_b_job_uid,
                "params": job_params,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "lane": used_lane
            }

            # 6. Optional Wait for Completion
            if wait_for_completion:
                print(f"⏳ Waiting for FSC validation job {job.uid} to complete...")
                try:
                    final_status = self.wait_for_job_completion(
                        project_uid=project_uid,
                        job_uid=job.uid,
                        workspace_uid=workspace_uid,
                        timeout=timeout,
                        check_interval=check_interval
                    )
                    result["status"] = final_status["status"]
                    result["final_status"] = final_status

                    # Try to extract FSC results from the job
                    if final_status["status"] == "completed":
                        print(f"✅ FSC validation job {job.uid} completed successfully!")
                        try:
                            fsc_info = self.get_refinement_fsc_info(project_uid, job.uid)
                            if fsc_info.get("success"):
                                result["fsc_info"] = fsc_info
                        except Exception as fsc_error:
                            print(f"⚠️ Could not extract FSC info: {fsc_error}")
                    else:
                        print(f"⚠️ FSC validation job {job.uid} finished with status: {final_status['status']}")
                except TimeoutError:
                    result["status"] = "timeout"
                    print(f"⏰ FSC validation job {job.uid} timed out after {timeout} seconds")
                except Exception as e:
                    result["status"] = "error"
                    print(f"❌ Error monitoring FSC validation job {job.uid}: {e}")

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "job_type": "compute_fsc_validation",
                "message": f"Failed to compute FSC validation: {str(e)}"
            }