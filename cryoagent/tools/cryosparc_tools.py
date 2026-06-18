"""CryoSPARC tools for cryoEM image processing."""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from cryosparc.tools import CryoSPARC
from ..config.config_loader import CryoSPARCSettings
from .cryosift_tools import CryoSiftTools, CryoSiftPaths


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

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        """
        Coerce a possibly-LLM-supplied value to a float, or return None.

        Returns None for None, empty strings, and non-numeric tokens such as
        "auto"/"default"/"none" so the caller simply omits the parameter and lets
        CryoSPARC use its own default, instead of forwarding an invalid value.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            token = value.strip().lower()
            if token in ("", "auto", "default", "none", "null", "na", "n/a"):
                return None
            try:
                return float(token)
            except ValueError:
                return None
        return None

    @staticmethod
    def _merge_passthrough_params(
        job_params: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Merge caller-supplied raw CryoSPARC parameters into job_params.

        Precedence (lowest to highest): existing job_params (friendly-name
        defaults set by the wrapper) < kwargs (legacy passthrough) < params
        (explicit raw-key dict the LLM controls). This lets the agent override
        any CryoSPARC parameter by its real key (discoverable via
        describe_job_params) while preserving the friendly-name conveniences.
        """
        merged = dict(job_params)
        if kwargs:
            merged.update({k: v for k, v in kwargs.items() if v is not None})
        if params:
            if not isinstance(params, dict):
                raise ValueError(
                    f"'params' must be a dict of CryoSPARC parameter keys, got {type(params).__name__}"
                )
            merged.update(params)
        return merged

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

    def _resolve_particles_slot(self, project, job_uid: str) -> str:
        """
        Resolve the correct particles output slot to connect FROM a given job.

        Inspects the job's real output_result_groups rather than guessing from the
        UID. Preference order when several particle groups exist:
          particles_selected > particles > particles_all_classes > first particle group.
        This avoids "No match for particles in job J###" when connecting from a
        select_2D job (which exposes particles_selected / particles_excluded, with
        no plain 'particles' group).
        """
        try:
            job = project.find_job(job_uid)
            job.refresh()
            doc = getattr(job, "doc", {}) or {}
            outputs = doc.get("output_result_groups", []) or []
            particle_groups = [
                g.get("name") for g in outputs
                if (g.get("type") == "particle") and g.get("name")
            ]
            if not particle_groups:
                # No particle outputs detected; fall back to the conventional name.
                return "particles"
            for preferred in ("particles_selected", "particles", "particles_all_classes"):
                if preferred in particle_groups:
                    return preferred
            # Otherwise use the first particle group that is not an "excluded" set.
            for name in particle_groups:
                if "exclud" not in name.lower():
                    return name
            return particle_groups[0]
        except Exception:
            return "particles"

    def _resolve_output_slot(self, project, job_uid: str, slot_type: str,
                             default: Optional[str] = None) -> Optional[str]:
        """Resolve the first output group of a given type (volume/mask/particle/
        exposure) to connect FROM a job, by inspecting its real output_result_groups.

        Returns the slot name, or `default` (or `slot_type` itself) when none found.
        Used by the resolution-improvement tools that consume volume/mask inputs.
        """
        fallback = default if default is not None else slot_type
        try:
            job = project.find_job(job_uid)
            job.refresh()
            doc = getattr(job, "doc", {}) or {}
            outputs = doc.get("output_result_groups", []) or []
            matches = [g.get("name") for g in outputs
                       if (g.get("type") == slot_type) and g.get("name")]
            if not matches:
                return fallback
            # Prefer the conventional bare name when present (e.g. 'volume', 'mask').
            if slot_type in matches:
                return slot_type
            # Prefer a sharpened/refined volume's main map over half-maps if named so.
            for preferred in (f"{slot_type}", "mask_refine", "volume"):
                if preferred in matches:
                    return preferred
            return matches[0]
        except Exception:
            return fallback

    def _queue_job_with_lane_fallback(
        self,
        job,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        log_prefix: str = "⚙️ No lane specified; retrying queue on lane",
        logger=None,
    ) -> Optional[str]:
        """Queue a CryoSPARC job with shared lane fallback behavior.

        Returns:
            The lane that was explicitly used, or None when queueing succeeded
            without a lane.
        """
        configured_lane = getattr(self.settings, "lane", None)
        requested_lane = lane if lane is not None else configured_lane
        used_lane = requested_lane
        try:
            job.queue(lane=requested_lane, hostname=hostname)
        except Exception as queue_error:
            message = str(queue_error)
            if (requested_lane is None and hostname is None and "Must specify a lane" in message):
                try:
                    lanes = self.cs.get_lanes()
                    if not lanes:
                        raise queue_error
                    used_lane = lanes[0]["name"]
                    log_message = f"{log_prefix} '{used_lane}'"
                    if logger is not None:
                        logger.info(log_message)
                    else:
                        print(log_message)
                    job.queue(lane=used_lane)
                except Exception:
                    raise queue_error
            else:
                raise queue_error
        return used_lane

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

    def import_micrographs(
        self,
        project_uid: str,
        workspace_uid: str,
        micrographs_path: str,
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
        Import micrographs directly into CryoSPARC (skips motion correction).
        
        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            micrographs_path: Path to micrograph files
            pixel_size: Pixel size in Angstroms
            voltage: Acceleration voltage in kV
            cs_mm: Spherical aberration in mm
            dose: Total electron dose in e-/Å²
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            check_interval: Interval between status checks in seconds
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
                "blob_paths": micrographs_path,
                "psize_A": pixel_size,
                "accel_kv": voltage,
                "cs_mm": cs_mm,
                "total_dose_e_per_A2": dose,
            }
            
            if kwargs:
                job_params.update(kwargs)
            
            # Create job using workspace.create_job()
            job = workspace.create_job("import_micrographs", params=job_params)
            
            # Queue the job (handle lane requirement if needed)
            self._queue_job_with_lane_fallback(
                job,
                log_prefix="No lane specified; using default lane",
            )
            
            print(f"Queued import micrographs job: {job.uid}")
            
            self._job_cache[job.uid] = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            result = {
                "job_uid": job.uid,
                "job_type": "import_micrographs",
                "status": "queued",
                "params": job_params,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            
            # Wait for completion if requested
            if wait_for_completion:
                print(f"⏳ Waiting for import micrographs job {job.uid} to complete...")
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
                        print(f"✅ Import micrographs job {job.uid} completed successfully!")
                    else:
                        print(f"⚠️ Import micrographs job {job.uid} finished with status: {final_status['status']}")
                except TimeoutError:
                    result["status"] = "timeout"
                    print(f"⏰ Import micrographs job {job.uid} timed out after {timeout} seconds")
                except Exception as e:
                    result["status"] = "error"
                    print(f"❌ Error monitoring import micrographs job {job.uid}: {e}")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to import micrographs: {e}")

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

            used_lane = self._queue_job_with_lane_fallback(
                job,
                lane=lane,
                hostname=hostname,
            )
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
    
    def _resolve_import_movies_output_labels(self, project, movies_job_uid: str) -> List[str]:
        """Return output labels with items for an import movies job."""
        available_output_labels: List[str] = []
        try:
            import_job = project.find_job(movies_job_uid)
            import_job.refresh()
            job_doc = getattr(import_job, "doc", {})
            job_status = job_doc.get("status", "unknown")

            output_result_groups = job_doc.get("output_result_groups", [])
            for group in output_result_groups:
                label = group.get("name")
                num_items = group.get("num_items", 0)
                if label and num_items > 0:
                    available_output_labels.append(label)

            if not available_output_labels:
                raise RuntimeError(
                    f"Import movies job {movies_job_uid} (status: {job_status}) has no outputs. "
                    f"This usually means no movies were found at the specified path or the import failed. "
                    f"Please check the import job log and verify the movies_path is correct."
                )
        except RuntimeError:
            raise
        except Exception:
            return ["imported_movies", "movies"]

        return available_output_labels

    def _normalize_movies_job_uids(
        self,
        movies_job_uid: Optional[Union[str, List[str]]] = None,
        movies_job_uids: Optional[Union[str, List[str]]] = None,
    ) -> List[str]:
        """Normalize single or multiple import job UID arguments."""
        raw_values: List[str] = []
        for value in (movies_job_uids, movies_job_uid):
            if value is None:
                continue
            if isinstance(value, list):
                raw_values.extend(str(item).strip() for item in value if str(item).strip())
            else:
                raw_values.extend(
                    part.strip()
                    for part in str(value).split(",")
                    if part.strip()
                )

        deduped: List[str] = []
        seen = set()
        for uid in raw_values:
            if uid not in seen:
                seen.add(uid)
                deduped.append(uid)
        return deduped

    def motion_correction(
        self,
        project_uid: str,
        workspace_uid: str,
        movies_job_uid: Optional[Union[str, List[str]]] = None,
        movies_job_uids: Optional[Union[str, List[str]]] = None,
        binning: Optional[int] = None,
        patch_size: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
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
            movies_job_uid: UID(s) of one or more import movies jobs
            movies_job_uids: Alias for movies_job_uid when connecting multiple imports
            binning: Deprecated/no-op for patch_motion_correction_multi (kept for
                backward compatibility; this job type has no binning parameter).
                Use the `params` dict with real keys (e.g. output_fcrop_factor) instead.
            patch_size: Deprecated/no-op for patch_motion_correction_multi (kept for
                backward compatibility; use `params` with override_K_X/Y/Z instead).
            params: Raw CryoSPARC parameter dict forwarded verbatim to create_job
                (e.g. {"res_max_align": 5, "bfactor": 500}). Discover valid keys
                with describe_job_params("motion_correction"). Takes precedence.
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            **kwargs: Additional raw CryoSPARC parameters (legacy passthrough)

        Returns:
            Dictionary containing job information
        """
        try:
            normalized_job_uids = self._normalize_movies_job_uids(
                movies_job_uid=movies_job_uid,
                movies_job_uids=movies_job_uids,
            )
            if not normalized_job_uids:
                raise RuntimeError(
                    "At least one import movies job UID is required for motion correction."
                )

            # Find project and workspace
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)

            # patch_motion_correction_multi has no binning/patch_size parameters;
            # warn if a caller still passes them so the intent isn't silently lost.
            if binning is not None:
                print(
                    "⚠️ motion_correction: 'binning' is not a parameter of "
                    "patch_motion_correction_multi and will be ignored. Use the "
                    "params dict (see describe_job_params) for real keys."
                )
            if patch_size is not None:
                print(
                    "⚠️ motion_correction: 'patch_size' is not a parameter of "
                    "patch_motion_correction_multi and will be ignored. Use the "
                    "params dict (see describe_job_params) for real keys."
                )

            job_params = self._merge_passthrough_params({}, params=params, kwargs=kwargs)

            per_job_output_labels: List[List[str]] = []
            for import_job_uid in normalized_job_uids:
                per_job_output_labels.append(
                    self._resolve_import_movies_output_labels(project, import_job_uid)
                )

            connection_errors: List[Tuple[str, Exception]] = []
            job = None
            selected_connections: List[Tuple[str, str]] = []

            if len(normalized_job_uids) == 1:
                output_labels_to_try = per_job_output_labels[0]
                for output_label in output_labels_to_try:
                    try:
                        job = workspace.create_job(
                            "patch_motion_correction_multi",
                            params=job_params,
                            connections={"movies": (normalized_job_uids[0], output_label)},
                        )
                        selected_connections = [(normalized_job_uids[0], output_label)]
                        break
                    except Exception as exc:
                        connection_errors.append((output_label, exc))
                        job = None
            else:
                candidate_labels = [
                    labels[0] if labels else "imported_movies"
                    for labels in per_job_output_labels
                ]
                label_variants = [
                    candidate_labels,
                    ["imported_movies"] * len(normalized_job_uids),
                    ["movies"] * len(normalized_job_uids),
                ]
                for labels in label_variants:
                    movie_connections = list(zip(normalized_job_uids, labels))
                    try:
                        job = workspace.create_job(
                            "patch_motion_correction_multi",
                            params=job_params,
                            connections={"movies": movie_connections},
                        )
                        selected_connections = movie_connections
                        break
                    except Exception as exc:
                        connection_errors.append((",".join(labels), exc))
                        job = None

            if job is None:
                error_messages = ", ".join(
                    f"output '{label}': {err}" for label, err in connection_errors
                ) or "unknown"
                raise RuntimeError(
                    "Unable to connect motion correction to import job outputs: "
                    f"{error_messages}. "
                    f"Import jobs {normalized_job_uids} may not have produced valid outputs. "
                    f"Please verify the import jobs completed successfully and check their logs."
                )

            # Queue the job
            used_lane = self._queue_job_with_lane_fallback(
                job,
                lane=lane,
                hostname=hostname,
            )
            print(f"Queued motion correction job: {job.uid}")

            self._job_cache[job.uid] = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            movies_connection_value: Union[str, List[Tuple[str, str]]]
            if len(selected_connections) == 1:
                movies_connection_value = selected_connections[0][0]
            else:
                movies_connection_value = selected_connections

            result = {
                "job_uid": job.uid,
                "job_type": "patch_motion_correction_multi",
                "status": "queued",
                "params": job_params,
                "connections": {"movies": movies_connection_value},
                "movies_job_uids": normalized_job_uids,
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
        group_job_uid: Optional[str] = None,
        min_res: Optional[float] = None,
        max_res: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
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
            min_res: Minimum resolution for CTF fit search in Angstroms. Mapped to
                the real CryoSPARC key `res_min_align` (default 25). Only applied
                when provided.
            max_res: Maximum resolution for CTF fit search in Angstroms. Mapped to
                the real CryoSPARC key `res_max_align` (default 4). Only applied
                when provided.
            params: Raw CryoSPARC parameter dict forwarded verbatim to create_job
                (e.g. {"df_search_max": 50000, "amp_contrast": 0.1}). Discover
                valid keys with describe_job_params("ctf_estimation"). Takes precedence.
            wait_for_completion: Whether to wait for job completion
            timeout: Maximum time to wait for completion in seconds
            **kwargs: Additional raw CryoSPARC parameters (legacy passthrough)

        Returns:
            Dictionary containing job information
        """
        try:
            if not micrographs_job_uid or not isinstance(micrographs_job_uid, str):
                raise ValueError(
                    "micrographs_job_uid is required and must be a non-empty string (e.g. job UID from motion correction like J4). "
                    f"Got: {type(micrographs_job_uid).__name__} = {micrographs_job_uid!r}"
                )
            # Find project and workspace
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)

            # Map friendly resolution names to the real CryoSPARC keys.
            job_params: Dict[str, Any] = {}
            if min_res is not None:
                job_params["res_min_align"] = min_res
            if max_res is not None:
                job_params["res_max_align"] = max_res

            job_params = self._merge_passthrough_params(job_params, params=params, kwargs=kwargs)

            if group_job_uid is None:
                group_job_uid = "micrographs"


            # Create job with connections - use job UID directly for connections
            job = workspace.create_job(
                "patch_ctf_estimation_multi",
                params=job_params,
                connections={"exposures": (micrographs_job_uid, group_job_uid)}
            )
            
            # Queue the job
            used_lane = self._queue_job_with_lane_fallback(
                job,
                lane=lane,
                hostname=hostname,
            )
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
                print()  # Print newline when job completes
                return status
            
            progress_display = f"{status['progress']}%" if status.get("progress") is not None else "N/A"
            message = status.get("message")
            line = f"Job {job_uid} status: {status['status']} ({progress_display})"
            if message:
                line += f" - {message}"
            print(f"\r{line}", end='', flush=True)
            time.sleep(check_interval)
        
        print()  # Print newline before timeout error
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

    # ------------------------------------------------------------------
    # Parameter introspection
    # ------------------------------------------------------------------

    # Map the friendly job-type names the agents use to the real CryoSPARC
    # job-type identifiers so introspection works with either spelling.
    _JOB_TYPE_ALIASES: Dict[str, str] = {
        "import_movies": "import_movies",
        "import_micrographs": "import_micrographs",
        "import_particles": "import_particles",
        "motion_correction": "patch_motion_correction_multi",
        "patch_motion": "patch_motion_correction_multi",
        "ctf_estimation": "patch_ctf_estimation_multi",
        "patch_ctf": "patch_ctf_estimation_multi",
        "micrograph_selection": "curate_exposures_v2",
        "curate_exposures": "curate_exposures_v2",
        "blob_picker": "blob_picker_gpu",
        "template_picker": "template_picker_gpu",
        "extract_particles": "extract_micrographs_multi",
        "extract": "extract_micrographs_multi",
        "class_2d": "class_2D",
        "class2d": "class_2D",
        "select_2d_classes": "select_2D",
        "select_2d": "select_2D",
        "ab_initio_reconstruction": "homo_abinit",
        "ab_initio": "homo_abinit",
        "abinit": "homo_abinit",
        "homogeneous_refinement": "homo_refine_new",
        "homo_refine": "homo_refine_new",
        "nonuniform_refinement": "nonuniform_refine_new",
        "nonuniform_refine": "nonuniform_refine_new",
        "heterogeneous_refinement": "hetero_refine",
        "hetero_refine": "hetero_refine",
        "reference_motion_correction": "reference_motion_correction",
        # SPA resolution-improvement + deep-picking tools.
        "ctf_refine_global": "ctf_refine_global",
        "ctf_refine_local": "ctf_refine_local",
        "local_refinement": "new_local_refine",
        "new_local_refine": "new_local_refine",
        "particle_subtract": "particle_subtract",
        "symmetry_expansion": "sym_expand",
        "sym_expand": "sym_expand",
        "sharpen": "sharpen",
        "deepemhancer": "deepemhancer",
        "local_resolution": "local_resolution",
        "class_3d": "class_3D",
        "class_3D": "class_3D",
        "variability_3d": "var_3D",
        "var_3D": "var_3D",
        "remove_duplicate_particles": "remove_duplicate_particles",
        "downsample_particles": "downsample_particles",
        "topaz_train": "topaz_train",
        "topaz_extract": "topaz_extract",
        "topaz_denoise": "topaz_denoise",
        "class_2d_new": "class_2D_new",
        "class_2D_new": "class_2D_new",
    }

    def _resolve_job_type(self, job_type: str) -> str:
        """Resolve a friendly or raw job-type name to the real CryoSPARC identifier."""
        if not job_type:
            return job_type
        key = str(job_type).strip()
        return self._JOB_TYPE_ALIASES.get(key, self._JOB_TYPE_ALIASES.get(key.lower(), key))

    def describe_job_params(
        self,
        job_type: str,
        include_hidden: bool = False,
        project_uid: Optional[str] = None,
        workspace_uid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return the full parameter specification for a CryoSPARC job type.

        Reads the connected instance's job specs (no job is created), so the
        caller can discover the exact raw parameter keys, types and defaults that
        a given job type accepts before submitting it via the generic ``params``
        passthrough on the corresponding tool.

        Uses the documented cryosparc-tools introspection data: the list of job
        sections, each containing JobSpec entries with a ``params_base`` mapping.
        On clients that expose ``CryoSPARC.get_job_specs`` (>= 4.5) that method is
        used; on older clients (e.g. 4.3.x) the same data is fetched directly from
        the command server via ``cli.get_config_var("job_types_available")``,
        which is exactly what get_job_specs wraps.

        Args:
            job_type: A friendly name (e.g. "motion_correction", "class_2d") or
                a raw CryoSPARC job-type id (e.g. "patch_motion_correction_multi").
            include_hidden: When True, include parameters CryoSPARC marks hidden.
            project_uid, workspace_uid: Accepted for API compatibility; not needed
                (no job is created).

        Returns:
            Dictionary with the resolved job_type and a "params" mapping of
            {param_key: {title, type, default}}.
        """
        resolved = self._resolve_job_type(job_type)
        specs = self._fetch_job_specs()

        params_base = None
        spec_title = None
        for section in specs:
            for job_spec in (section.get("contains", []) or []):
                if job_spec.get("name") == resolved:
                    params_base = job_spec.get("params_base", {}) or {}
                    spec_title = job_spec.get("title")
                    break
            if params_base is not None:
                break

        if params_base is None:
            available = sorted(
                js.get("name")
                for section in specs
                for js in (section.get("contains", []) or [])
                if js.get("name")
            )
            raise ValueError(
                f"Unknown job type '{job_type}' (resolved to '{resolved}'). "
                f"Available job types include: {', '.join(available[:60])}"
                + (" ..." if len(available) > 60 else "")
            )

        params: Dict[str, Any] = {}
        for key, details in params_base.items():
            if details.get("hidden") and not include_hidden:
                continue
            params[key] = {
                "title": details.get("title"),
                "type": details.get("type"),
                "default": details.get("value"),
            }

        return {
            "job_type": resolved,
            "requested_job_type": job_type,
            "title": spec_title,
            "param_count": len(params),
            "params": params,
        }

    def _fetch_job_specs(self) -> List[Dict[str, Any]]:
        """
        Fetch the instance's job specs (sections -> contains -> JobSpec with
        params_base) using the documented cryosparc-tools API.

        Prefers ``CryoSPARC.get_job_specs()`` when available; otherwise calls the
        same underlying command-server variable ``job_types_available`` directly
        via the CLI proxy (available on older clients such as 4.3.x).
        """
        # Preferred documented method (cryosparc-tools >= ~4.5).
        get_job_specs = getattr(self.cs, "get_job_specs", None)
        if callable(get_job_specs):
            try:
                specs = get_job_specs()
                if specs:
                    return list(specs)
            except Exception as e:
                print(f"⚠️  get_job_specs() failed ({e}); using cli.get_config_var fallback.")

        # Fallback: fetch the raw config variable that get_job_specs wraps. Works
        # on older clients that lack get_job_specs but still expose the CLI proxy.
        cli = getattr(self.cs, "cli", None)
        get_config_var = getattr(cli, "get_config_var", None) if cli is not None else None
        if callable(get_config_var):
            specs = get_config_var("job_types_available")
            if specs:
                return list(specs)
            raise RuntimeError(
                "CryoSPARC returned no 'job_types_available' specs; cannot introspect "
                "job parameters on this instance."
            )

        raise RuntimeError(
            "This CryoSPARC client exposes neither get_job_specs nor "
            "cli.get_config_var; cannot introspect job parameters."
        )

    def blob_picker(
        self,
        project_uid: str,
        workspace_uid: str,
        micrographs_job_uid: str,
        particle_diameter: float,
        diameter_max: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
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
            job_params = self._merge_passthrough_params(
                {"diameter": particle_diameter, "diameter_max": diameter_max},
                params=params,
                kwargs=kwargs,
            )
            
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
            used_lane = self._queue_job_with_lane_fallback(
                job,
                lane=lane,
                hostname=hostname,
            )
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
        params: Optional[Dict[str, Any]] = None,
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
            job_params = self._merge_passthrough_params(
                {"box_size_pix": box_size_pix},
                params=params,
                kwargs=kwargs,
            )
            
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
            used_lane = self._queue_job_with_lane_fallback(
                job,
                lane=lane,
                hostname=hostname,
            )
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
        force_max: Optional[bool] = None,
        batchsize_per_class: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
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
            force_max: If True, maximize over poses and shifts when aligning particles to references. 
                      If False, marginalize over poses and shifts to account for alignment uncertainty.
            batchsize_per_class: Number of particles per class to use during each iteration of online-EM.
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
            }
            
            # Add force_max parameter if provided
            if force_max is not None:
                job_params["class2D_force_max"] = force_max
            
            # Add batchsize_per_class parameter if provided
            if batchsize_per_class is not None:
                job_params["class2D_num_full_iter_batchsize_per_class"] = batchsize_per_class

            # Merge raw CryoSPARC parameter passthrough (LLM-controlled, takes precedence)
            job_params = self._merge_passthrough_params(job_params, params=params, kwargs=kwargs)
            
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
            used_lane = self._queue_job_with_lane_fallback(
                job,
                lane=lane,
                hostname=hostname,
            )
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
            lane: Ignored; select_2D is interactive and must queue on the master without a lane.
            hostname: Ignored; must be None for interactive queueing.
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
            
            # select_2D is interactive: CryoSPARC requires lane=None and hostname=None
            # (see validate_enqueue_job). Do not use settings.lane or lane fallback here.
            job.queue(lane=None, hostname=None)
            used_lane: Optional[str] = None
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
                        evaluator_script_path = cryosift_cfg.get("evaluator_script_path")
                        python_executable = cryosift_cfg.get("python_executable", "python")
                        conda_env = cryosift_cfg.get("conda_env")
                        extra_args = cryosift_cfg.get("extra_args")

                        # Create CryoSiftPaths if evaluator_script_path or weights_path is provided
                        cryosift_paths = None
                        if evaluator_script_path or weights_path:
                            # Use provided paths or fall back to defaults
                            default_weights_path = Path(weights_path) if weights_path else CryoSiftPaths().default_weights
                            evaluator_script = Path(evaluator_script_path) if evaluator_script_path else CryoSiftPaths().evaluator_script
                            cryosift_paths = CryoSiftPaths(
                                evaluator_script=evaluator_script,
                                default_weights=default_weights_path,
                            )

                        cryosift_tool = CryoSiftTools(
                            paths=cryosift_paths,
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
        params: Optional[Dict[str, Any]] = None,
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

            # Merge raw CryoSPARC parameter passthrough (LLM-controlled, takes
            # precedence). Previously a hardcoded whitelist silently dropped any
            # other param; now any valid template_picker_gpu key can be set.
            # blob_picker_job_uid is a helper arg, not a job param — exclude it.
            passthrough_kwargs = {
                k: v for k, v in kwargs.items() if k != "blob_picker_job_uid"
            }
            job_params = self._merge_passthrough_params(
                job_params, params=params, kwargs=passthrough_kwargs
            )

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
            used_lane = self._queue_job_with_lane_fallback(
                job,
                lane=lane,
                hostname=hostname,
            )
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

    # ------------------------------------------------------------------
    # Generic result introspection (for data-driven workflow)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_resolution_from_stats(stats: Dict[str, Any]) -> Optional[float]:
        """
        Pull the reported resolution in Angstroms from a latest_summary_stats dict.

        Handles the layouts seen across refinement types: resolution lives either
        directly under the stats, or nested under 'fsc_info'/'fsc_info_autotight'/
        'fsc_info_best'. The reported value is the loose-mask (or noise-substituted)
        resolution; only the '*_A' keys are in Angstroms (others are Fourier radius).
        Preference order favors the gold-standard masked estimate.
        """
        if not isinstance(stats, dict):
            return None
        # Candidate containers, in order of preference.
        containers = [stats]
        for key in ("fsc_info_best", "fsc_info", "fsc_info_autotight"):
            sub = stats.get(key)
            if isinstance(sub, dict):
                containers.append(sub)
        # Preferred Angstrom keys, best-practice first.
        pref_keys = (
            "radwn_loosemask_A",
            "radwn_noisesub_A",
            "radwn_tightmask_A",
            "radwn_sphericalmask_A",
            "radwn_final_A",
        )
        for container in containers:
            for k in pref_keys:
                v = container.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
        return None

    def describe_job_results(
        self,
        job_uid: str,
        project_uid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return a compact, factual summary of a completed job's results.

        Intended for data-driven workflow decisions: reports the real measured
        values (no canned classification beyond labeling the cFAR band). Read-only
        and cheap — does NOT create jobs. For 3D refinement jobs it reports
        resolution; cFAR is reported only if a downstream orientation_diagnostics
        job already exists (use get_orientation_diagnostics to compute it).

        Returns a dict with: job_uid, job_type, status, and (when available)
        resolution_angstroms, box_size, symmetry, num_particles, per-class info,
        cfar + cfar_label, plus output_groups (name/type/num_items).
        """
        try:
            cached = self._job_cache.get(job_uid, {})
            project_uid = project_uid or cached.get("project_uid")
            if not project_uid:
                raise ValueError(
                    "project_uid is required to describe job results "
                    "(pass it or ensure the job was queued via CryoSPARCTools)."
                )

            job = self.cs.find_job(project_uid, job_uid)
            job.refresh()
            doc = getattr(job, "doc", {}) or {}
            job_type = doc.get("type") or doc.get("job_type")
            status = doc.get("status")
            output_groups = doc.get("output_result_groups", []) or []

            result: Dict[str, Any] = {
                "success": True,
                "job_uid": job_uid,
                "job_type": job_type,
                "status": status,
            }

            # Compact list of output groups (name/type/count) — useful provenance.
            result["output_groups"] = [
                {
                    "name": g.get("name"),
                    "type": g.get("type"),
                    "num_items": g.get("num_items"),
                }
                for g in output_groups if isinstance(g, dict)
            ]

            # Total particles: prefer a 'particles' / 'particles_all_classes' group.
            for g in output_groups:
                if isinstance(g, dict) and g.get("type") == "particle":
                    name = g.get("name") or ""
                    if name in ("particles", "particles_all_classes", "particles_selected"):
                        result["num_particles"] = g.get("num_items")
                        break

            # Symmetry actually used, if recorded.
            params_spec = doc.get("params_spec", {}) or {}
            for sk in ("refine_symmetry", "abinit_symmetry", "multirefine_symmetry"):
                if sk in params_spec and isinstance(params_spec[sk], dict):
                    result["symmetry"] = params_spec[sk].get("value")
                    break

            jt = (job_type or "").lower()
            is_hetero = "hetero" in jt
            is_refine = "refine" in jt or "abinit" in jt

            if is_hetero:
                # Per-class resolution + particle distribution.
                cls = self.get_heterogeneous_refinement_class_resolutions(project_uid, job_uid)
                classes = cls.get("classes", []) if cls.get("success") else []
                # Attach per-class particle counts from particles_class_<i> groups.
                counts = {}
                for g in output_groups:
                    if isinstance(g, dict) and (g.get("name") or "").startswith("particles_class_"):
                        try:
                            cid = int((g["name"]).replace("particles_class_", ""))
                            counts[cid] = g.get("num_items")
                        except (ValueError, TypeError):
                            continue
                total = sum(v for v in counts.values() if isinstance(v, (int, float))) or 0
                for c in classes:
                    cid = c.get("class_id")
                    c["num_particles"] = counts.get(cid)
                    c["particle_fraction"] = (
                        round(counts[cid] / total, 4) if total and counts.get(cid) is not None else None
                    )
                    # Read EXISTING per-class cFAR only (do not compute here — the LLM
                    # decides which class is worth running orientation diagnostics on).
                    vol_slot = c.get("group_name") or f"volume_class_{cid}"
                    od = self._find_orientation_diagnostics(project_uid, job_uid, volume_group_name=vol_slot)
                    if od is not None:
                        c["cfar"] = od.get("cfar")
                        c["cfar_label"] = self._cfar_label(od.get("cfar"))
                        c["orientation_diagnostics_job_uid"] = od.get("job_uid")
                    else:
                        c["cfar"] = None
                result["classes"] = classes
                result["num_classes"] = len(classes)
                if any(c.get("cfar") is None for c in classes):
                    result["cfar_note"] = (
                        "Per-class cFAR not computed for some classes. Use "
                        "get_orientation_diagnostics with volume_group_name/particles_group_name "
                        "(e.g. volume_class_1 / particles_class_1) to compute cFAR for a class worth evaluating."
                    )
            elif is_refine:
                # Single-volume resolution from latest_summary_stats.
                res = None
                box = None
                for g in output_groups:
                    if isinstance(g, dict) and g.get("latest_summary_stats"):
                        stats = g["latest_summary_stats"]
                        res = self._extract_resolution_from_stats(stats)
                        # box size N if present
                        for cont in (stats, stats.get("fsc_info", {}), stats.get("fsc_info_autotight", {})):
                            if isinstance(cont, dict) and cont.get("N"):
                                box = cont.get("N"); break
                        if res is not None:
                            break
                if res is not None:
                    result["resolution_angstroms"] = res
                if box is not None:
                    result["box_size"] = int(box)
                # cFAR only if a downstream orientation_diagnostics job already exists.
                od = self._find_orientation_diagnostics(project_uid, job_uid)
                if od is not None:
                    result["cfar"] = od.get("cfar")
                    result["cfar_label"] = self._cfar_label(od.get("cfar"))
                    result["orientation_diagnostics_job_uid"] = od.get("job_uid")
                else:
                    result["cfar"] = None
                    result["cfar_note"] = "Run get_orientation_diagnostics to compute cFAR for this refinement."

            return result

        except Exception as e:
            return {
                "success": False,
                "job_uid": job_uid,
                "error": f"Failed to describe job results for {job_uid}: {str(e)}",
            }

    @staticmethod
    def _cfar_label(cfar: Optional[float]) -> Optional[str]:
        """Label a cFAR value per the agreed bands (reporting only, not a gate)."""
        if cfar is None:
            return None
        try:
            c = float(cfar)
        except (TypeError, ValueError):
            return None
        if c > 0.5:
            return "good"
        if c >= 0.15:
            return "acceptable (not ideal)"
        if c >= 0.1:
            return "poor"
        return "very poor — likely no real structure or severe preferred orientation"

    @staticmethod
    def _extract_cfar_from_job(job) -> Optional[float]:
        """
        Extract the cFAR (conical FSC area ratio) value from an
        orientation_diagnostics job's doc/summary stats.

        The exact key is confirmed at runtime against the live job; we search a
        set of likely keys recursively. cFAR is a single scalar in [0,1].
        """
        import re
        doc = getattr(job, "doc", {}) or {}

        def _search(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(v, (int, float)) and re.search(r"cfar|conical_fsc_area|area_ratio", str(k), re.I):
                        return float(v)
                for v in o.values():
                    r = _search(v)
                    if r is not None:
                        return r
            elif isinstance(o, list):
                for x in o:
                    r = _search(x)
                    if r is not None:
                        return r
            return None

        return _search(doc)

    def _find_orientation_diagnostics(
        self,
        project_uid: str,
        refinement_job_uid: str,
        volume_group_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Find an existing completed orientation_diagnostics job whose input is the
        given refinement job. When volume_group_name is given (e.g. a specific
        'volume_class_2' of a heterogeneous job), only match a diagnostics job that
        was connected to THAT volume slot — so per-class cFARs aren't confused.
        Returns {job_uid, cfar} or None.
        """
        try:
            refine_job = self.cs.find_job(project_uid, refinement_job_uid)
            refine_job.refresh()
            child_uids = refine_job.doc.get("children", []) or []
            for child_uid in child_uids:
                try:
                    child = self.cs.find_job(project_uid, child_uid)
                    child.refresh()
                    cdoc = child.doc
                    if cdoc.get("type") != "orientation_diagnostics" or cdoc.get("status") != "completed":
                        continue
                    if volume_group_name is not None:
                        # Only accept if this diagnostics job's volume input came from
                        # the requested slot of the refinement job.
                        if not self._diagnostics_uses_volume_slot(cdoc, refinement_job_uid, volume_group_name):
                            continue
                    return {"job_uid": child_uid, "cfar": self._extract_cfar_from_job(child)}
                except Exception:
                    continue
        except Exception:
            pass
        return None

    @staticmethod
    def _diagnostics_uses_volume_slot(diag_doc: Dict[str, Any], source_job_uid: str, volume_group_name: str) -> bool:
        """Check whether a diagnostics job's volume input connects to source_job_uid.volume_group_name."""
        try:
            for ig in diag_doc.get("input_slot_groups", []) or []:
                if (ig.get("type") or "") != "volume" and "volume" not in (ig.get("name") or ""):
                    continue
                for conn in ig.get("connections", []) or []:
                    juid = conn.get("job_uid") or conn.get("group_name")
                    gname = conn.get("group_name") or conn.get("slot_name")
                    if juid == source_job_uid and gname == volume_group_name:
                        return True
        except Exception:
            pass
        return False

    def get_orientation_diagnostics(
        self,
        project_uid: str,
        workspace_uid: str,
        refinement_job_uid: str,
        volume_group_name: Optional[str] = None,
        particles_group_name: Optional[str] = None,
        run_if_missing: bool = True,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get the cFAR (conical FSC area ratio) for a refinement, via an
        orientation_diagnostics job.

        First looks for an existing completed orientation_diagnostics job
        downstream of the refinement. If none exists and run_if_missing is True,
        creates and runs one, then reads cFAR.

        For a HETEROGENEOUS refinement, pass volume_group_name (e.g. "volume_class_1")
        and particles_group_name (e.g. "particles_class_1") to evaluate ONE class.
        When omitted, the first volume/particle output slots are used (appropriate
        for single-volume refinements).

        Returns: {success, cfar, cfar_label, job_uid, computed (bool)}.
        """
        try:
            existing = self._find_orientation_diagnostics(
                project_uid, refinement_job_uid, volume_group_name=volume_group_name
            )
            if existing is not None:
                return {
                    "success": True,
                    "cfar": existing.get("cfar"),
                    "cfar_label": self._cfar_label(existing.get("cfar")),
                    "job_uid": existing.get("job_uid"),
                    "computed": False,
                }

            if not run_if_missing:
                return {
                    "success": False,
                    "error": "No orientation_diagnostics job found; pass run_if_missing=True to compute.",
                }

            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)

            # Resolve the volume + particle output slots to connect.
            refine_job = project.find_job(refinement_job_uid)
            refine_job.refresh()
            rdoc = getattr(refine_job, "doc", {}) or {}
            vol_slot = volume_group_name
            part_slot = particles_group_name
            if vol_slot is None or part_slot is None:
                for g in rdoc.get("output_result_groups", []):
                    if not isinstance(g, dict):
                        continue
                    gt, gn = g.get("type"), g.get("name")
                    if gt == "volume" and vol_slot is None:
                        vol_slot = gn
                    if gt == "particle" and part_slot is None:
                        part_slot = gn
            vol_slot = vol_slot or "volume"
            part_slot = part_slot or "particles"

            # Orientation diagnostics requires half-maps (map_half_A / map_half_B) on
            # the volume input — cFAR is derived from the directional half-map FSC.
            # Heterogeneous-refinement class volumes (volume_class_i) expose only
            # 'map'/'map_sharp' (no half-maps), so cFAR cannot be computed directly on
            # a class. Detect this and return an actionable message instead of letting
            # the job fail cryptically.
            for g in rdoc.get("output_result_groups", []):
                if isinstance(g, dict) and g.get("name") == vol_slot:
                    slots = g.get("contains") or g.get("slots") or []
                    slot_names = {s.get("name") for s in slots if isinstance(s, dict)}
                    if slot_names and not {"map_half_A", "map_half_B"} <= slot_names:
                        return {
                            "success": False,
                            "error": (
                                f"Volume '{vol_slot}' of {refinement_job_uid} has no half-maps "
                                f"(map_half_A/map_half_B), so cFAR cannot be computed on it. "
                                f"This is expected for heterogeneous-refinement class volumes. "
                                f"To get cFAR for this class, first run a homogeneous/non-uniform "
                                f"refinement on its particles ({part_slot}) to produce half-maps, "
                                f"then run orientation diagnostics on that refinement."
                            ),
                        }
                    break

            connections = {
                "volume": (refinement_job_uid, vol_slot),
                "particles": (refinement_job_uid, part_slot),
            }
            job = workspace.create_job(
                "orientation_diagnostics",
                connections=connections,
                params=dict(kwargs) if kwargs else {},
            )
            self._queue_job_with_lane_fallback(job, lane=lane, hostname=hostname)
            print(f"Queued orientation diagnostics job: {job.uid} "
                  f"(volume={vol_slot}, particles={part_slot})")

            final = self.wait_for_job_completion(project_uid, job.uid, workspace_uid, timeout, check_interval)
            if final.get("status") != "completed":
                return {
                    "success": False,
                    "job_uid": job.uid,
                    "error": f"orientation_diagnostics finished with status: {final.get('status')}",
                }
            job.refresh()
            cfar = self._extract_cfar_from_job(job)
            return {
                "success": True,
                "cfar": cfar,
                "cfar_label": self._cfar_label(cfar),
                "job_uid": job.uid,
                "computed": True,
                "volume_group_name": vol_slot,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get orientation diagnostics for {refinement_job_uid}: {str(e)}",
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
        params: Optional[Dict[str, Any]] = None,
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
            if symmetry and str(symmetry).upper() != "C1":
                job_params["abinit_symmetry"] = symmetry

            # Merge raw CryoSPARC parameter passthrough (LLM-controlled, takes precedence)
            job_params = self._merge_passthrough_params(job_params, params=params, kwargs=kwargs)

            particles_output_slot = self._infer_particles_output_slot(project, particles_job_uid)
            
            job = workspace.create_job(
                "homo_abinit",  # Ab initio reconstruction job type
                connections={
                    "particles": (particles_job_uid, particles_output_slot)
                },
                params=job_params
            )
            
            # Queue the job with lane auto-detection
            used_lane = self._queue_job_with_lane_fallback(
                job,
                lane=lane,
                hostname=hostname,
            )
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
        refine_defocus_refine: bool = True,
        refine_ctf_global_refine: bool = True,
        params: Optional[Dict[str, Any]] = None,
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
        
        Note: For homogeneous refinement, particles and volume should come from different jobs:
        - particles_job_uid: From Select 2D job or import particle job
        - volume_job_uid: From ab initio reconstruction job
        The connections used are:
        - particles: (particles_job_uid, particles_output_slot)
        - volume: (volume_job_uid, volume_output_slot)
        
        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            particles_job_uid: UID of particles job (from Select 2D job or import particle job)
            volume_job_uid: UID of the ab initio reconstruction job (provides the initial volume)
            refinement_resolution: Target resolution in Angstroms (optional)
            symmetry: Symmetry group (e.g., C1, C2, D7) (default: C1)
            # Advanced refinement parameters
            refine_do_init_scale_est: Enable initial scale estimation (default: True)
            refine_highpass_res: High-pass filter resolution in Angstroms (optional)
            refine_num_final_iterations: Number of final refinement iterations (optional)
            refine_res_init: Initial resolution for refinement in Angstroms (optional)
            refine_symmetry_do_align: Enable symmetry alignment (default: True)
            refine_defocus_refine: Enable defocus refinement during CTF refinement (default: True)
            refine_ctf_global_refine: Enable global CTF refinement (default: True)
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
                # For homogeneous refinement: particles come from particles_job_uid (Select 2D/import), volume from volume_job_uid (ab initio)
                # They should be different - particles from 2D selection/extraction, volume from ab initio reconstruction
                
                # Check if particles_job_uid is different from volume_job_uid
                use_separate_particles = (particles_job_uid != volume_job_uid)
                
                if use_separate_particles:
                    # Homogeneous refinement case: particles from Select 2D/import job, volume from ab initio job
                    # Find particles output slot from particles job (Select 2D or import particle job)
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
                        print(f"⚠️  Could not detect particles slot from particles job, using default: {e}")
                    
                    # Find volume output slot from volume job (ab initio reconstruction job)
                    try:
                        volume_job = project.find_job(volume_job_uid)
                        volume_job.refresh()
                        volume_doc = getattr(volume_job, "doc", {})
                        volume_outputs = volume_doc.get("output_result_groups", [])
                        # Find volume output slot
                        volume_slot = "volume_class_0"  # Default for ab initio
                        for group in volume_outputs:
                            if group.get("type") == "volume":
                                volume_slot = group.get("name", "volume_class_0")
                                break
                    except Exception as e:
                        volume_slot = "volume_class_0"
                        print(f"⚠️  Could not detect volume slot from ab initio job, using default: {e}")
                    
                    print(f"ℹ️  Homogeneous refinement: particles from {particles_job_uid}.{particles_slot} (Select 2D/import), volume from {volume_job_uid}.{volume_slot} (ab initio)")
                else:
                    # Fallback case: both from same job (should be avoided, but kept for compatibility)
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
            
            # Add CTF refinement parameters (use explicit parameters, fallback to kwargs)
            refine_defocus_refine_value = kwargs.get("refine_defocus_refine", refine_defocus_refine)
            refine_ctf_global_refine_value = kwargs.get("refine_ctf_global_refine", refine_ctf_global_refine)
            job_params["refine_defocus_refine"] = refine_defocus_refine_value
            job_params["refine_ctf_global_refine"] = refine_ctf_global_refine_value
            
            # Add refinement resolution if specified. The friendly
            # 'refinement_resolution' maps to the real key 'refine_res_init'
            # (there is no 'refine_res' parameter). Non-numeric values like "auto"
            # are ignored so CryoSPARC uses its own default.
            _refine_res = self._coerce_float(refinement_resolution)
            if _refine_res is not None:
                job_params["refine_res_init"] = _refine_res

            # Add symmetry if specified and not C1
            if symmetry and symmetry != "C1":
                job_params["refine_symmetry"] = symmetry

            # Add high-pass filter resolution if specified
            _highpass = self._coerce_float(refine_highpass_res)
            if _highpass is not None:
                job_params["refine_highpass_res"] = _highpass

            # Add number of final iterations if specified
            if refine_num_final_iterations is not None:
                job_params["refine_num_final_iterations"] = refine_num_final_iterations

            # Add initial resolution if explicitly provided via refine_res_init.
            _res_init = self._coerce_float(refine_res_init)
            if _res_init is not None:
                job_params["refine_res_init"] = _res_init

            # Merge raw CryoSPARC parameter passthrough (LLM-controlled, takes precedence).
            # Exclude connection-slot hints which are not job parameters.
            passthrough_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ("particles_group_name", "volume_group_name")
            }
            job_params = self._merge_passthrough_params(
                job_params, params=params, kwargs=passthrough_kwargs
            )

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
            used_lane = self._queue_job_with_lane_fallback(
                job,
                lane=lane,
                hostname=hostname,
            )
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
    
    def nonuniform_refine_new(
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
        refine_defocus_refine: bool = True,
        refine_ctf_global_refine: bool = True,
        particles_group_names: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        # Job control parameters
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run non-uniform refinement to refine a single 3D structure.
        
        Note: For non-uniform refinement, both particles and volume come from the ab initio job.
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
            refine_defocus_refine: Enable defocus refinement during CTF refinement (default: True)
            refine_ctf_global_refine: Enable global CTF refinement (default: True)
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
            
            # Create non-uniform refinement job with comprehensive parameters
            job_params: Dict[str, Any] = {
                "refine_do_init_scale_est": refine_do_init_scale_est,
                "refine_symmetry_do_align": refine_symmetry_do_align
            }
            
            # Add CTF refinement parameters (use explicit parameters, fallback to kwargs)
            refine_defocus_refine_value = kwargs.get("refine_defocus_refine", refine_defocus_refine)
            refine_ctf_global_refine_value = kwargs.get("refine_ctf_global_refine", refine_ctf_global_refine)
            job_params["refine_defocus_refine"] = refine_defocus_refine_value
            job_params["refine_ctf_global_refine"] = refine_ctf_global_refine_value
            
            # Add refinement resolution if specified. The friendly
            # 'refinement_resolution' maps to the real key 'refine_res_init'
            # (there is no 'refine_res' parameter). Non-numeric values like "auto"
            # are ignored so CryoSPARC uses its own default.
            _refine_res = self._coerce_float(refinement_resolution)
            if _refine_res is not None:
                job_params["refine_res_init"] = _refine_res

            # Add symmetry if specified and not C1
            if symmetry and symmetry != "C1":
                job_params["refine_symmetry"] = symmetry

            # Add high-pass filter resolution if specified
            _highpass = self._coerce_float(refine_highpass_res)
            if _highpass is not None:
                job_params["refine_highpass_res"] = _highpass

            # Add number of final iterations if specified
            if refine_num_final_iterations is not None:
                job_params["refine_num_final_iterations"] = refine_num_final_iterations

            # Add initial resolution if explicitly provided via refine_res_init.
            _res_init = self._coerce_float(refine_res_init)
            if _res_init is not None:
                job_params["refine_res_init"] = _res_init

            # Merge raw CryoSPARC parameter passthrough (LLM-controlled, takes
            # precedence). Excludes connection-slot hints which are not job params.
            passthrough_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ("particles_group_name", "volume_group_name")
            }
            job_params = self._merge_passthrough_params(
                job_params, params=params, kwargs=passthrough_kwargs
            )

            # Create the job - non-uniform refinement job type
            # Support connecting multiple particle groups (e.g. several particles_class_X
            # outputs of one heterogeneous-refinement job) to the single "particles" input.
            if particles_group_names and len(particles_group_names) > 1:
                particles_connection = [
                    (particles_job_uid, group_name) for group_name in particles_group_names
                ]
                print(
                    f"ℹ️  Non-uniform refinement: connecting {len(particles_connection)} particle "
                    f"groups from {particles_job_uid}: {list(particles_group_names)}"
                )
            else:
                if particles_group_names and len(particles_group_names) == 1:
                    particles_slot = particles_group_names[0]
                particles_connection = (particles_job_uid, particles_slot)

            job = workspace.create_job(
                "nonuniform_refine_new",  # Non-uniform refinement job type
                connections={
                    "particles": particles_connection,
                    "volume": (volume_job_uid, volume_slot)
                },
                params=job_params
            )
            
            # Queue the job with lane auto-detection
            used_lane = self._queue_job_with_lane_fallback(
                job,
                lane=lane,
                hostname=hostname,
            )
            print(f"Queued non-uniform refinement job: {job.uid}")
            
            job_uid = job.uid
            
            result = {
                "success": True,
                "job_uid": job_uid,
                "job_type": "nonuniform_refine_new",
                "message": f"Non-uniform refinement job {job_uid} queued successfully",
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
                "job_type": "nonuniform_refine_new",
                "message": f"Failed to queue non-uniform refinement job: {str(e)}"
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
            
            job_params["mem_cache_sz"] = 0.95
            
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
            used_lane = self._queue_job_with_lane_fallback(
                job,
                lane=lane,
                hostname=hostname,
            )
            
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
        volume_job_uids: Optional[List[str]] = None,
        num_classes: Optional[int] = None,
        symmetry: str = "C1",
        volume_from_job_uid: Optional[str] = None,
        volume_group_names: Optional[List[str]] = None,
        particles_group_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run heterogeneous refinement to refine multiple 3D structures simultaneously.

        Two ways to supply the initial volumes:
        1. volume_job_uids: a list of separate volume job UIDs (each connected via its
           own volume output slot). The same job UID may be repeated K times to use one
           volume as K identical seeds.
        2. volume_from_job_uid (+ optional volume_group_names or num_classes): connect K
           distinct volume outputs of a SINGLE job, e.g. the volume_class_0..volume_class_{K-1}
           outputs of an ab initio / heterogeneous-refinement job. When volume_group_names
           is omitted, it defaults to [f"volume_class_{i}" for i in range(num_classes)].
        This second form replaces the manual workspace.create_job("hetero_refine") pattern
        previously used inside composite tools.

        Args:
            project_uid: CryoSPARC project UID
            workspace_uid: CryoSPARC workspace UID
            particles_job_uid: UID of the particles job
            volume_job_uids: List of volume job UIDs (from ab initio); mutually exclusive
                with volume_from_job_uid
            num_classes: Number of classes (default: inferred from the volume connections)
            symmetry: Symmetry group (e.g., C1, D7) - applied to all classes (default: C1)
            volume_from_job_uid: A single job whose K volume_class outputs seed the classes
            volume_group_names: Explicit volume output slots on volume_from_job_uid
            particles_group_name: Explicit particles output slot (else inferred)
            params: Raw CryoSPARC parameter passthrough merged into job params
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

            # Add symmetry if specified and not C1
            if symmetry and symmetry != "C1":
                job_params["multirefine_symmetry"] = symmetry

            job_params = self._merge_passthrough_params(job_params, params=params, kwargs=kwargs)

            # Resolve the particles connection slot by inspecting the job's ACTUAL
            # output groups (do NOT guess from the job-UID string). A select_2D job
            # exposes 'particles_selected'/'particles_excluded' (no plain 'particles'),
            # so connecting to 'particles' raises "No match for particles in job ...".
            if particles_group_name:
                particles_slot = particles_group_name
            else:
                particles_slot = self._resolve_particles_slot(project, particles_job_uid)

            # Build volume connections. Two modes:
            #  (A) volume_from_job_uid: connect K volume outputs of ONE job. The job may
            #      expose K distinct class volumes (e.g. an ab-initio job's
            #      volume_class_0..K-1) OR a single consensus volume (e.g. a homogeneous/
            #      non-uniform refinement's lone "volume" output). In the latter case the
            #      SAME volume is connected K times as identical seeds.
            #  (B) volume_job_uids: connect each listed job via its own volume slot.
            if volume_from_job_uid:
                if volume_group_names:
                    # Caller specified exact slots — trust them.
                    resolved_groups = list(volume_group_names)
                else:
                    if num_classes is None:
                        raise ValueError(
                            "heterogeneous_refinement: provide volume_group_names or num_classes "
                            "when using volume_from_job_uid."
                        )
                    k = int(num_classes)
                    # Inspect the source job's actual volume output slots.
                    available_volume_slots: List[str] = []
                    try:
                        src_job = project.find_job(volume_from_job_uid)
                        src_job.refresh()
                        src_doc = getattr(src_job, "doc", {})
                        for group in src_doc.get("output_result_groups", []):
                            if (group.get("type") == "volume") and group.get("name"):
                                available_volume_slots.append(group["name"])
                    except Exception as e:
                        print(f"⚠️  Could not inspect volume outputs of {volume_from_job_uid}: {e}")

                    class_slots = [s for s in available_volume_slots if s.startswith("volume_class_")]
                    if len(class_slots) >= k:
                        # Ab-initio / multi-class source: use its first K distinct class volumes.
                        resolved_groups = class_slots[:k]
                    elif len(available_volume_slots) == 1:
                        # Single consensus volume (refinement job): repeat it K times.
                        resolved_groups = [available_volume_slots[0]] * k
                    elif available_volume_slots:
                        # Some class volumes but fewer than K: pad by repeating the first.
                        resolved_groups = (class_slots or available_volume_slots)[:]
                        while len(resolved_groups) < k:
                            resolved_groups.append(resolved_groups[0])
                    else:
                        # Couldn't detect; fall back to the conventional single "volume" repeated.
                        resolved_groups = ["volume"] * k
                volume_connections = [(volume_from_job_uid, slot) for slot in resolved_groups]
                if num_classes is None:
                    num_classes = len(volume_connections)
                print(
                    f"ℹ️  Heterogeneous refinement: connecting {len(volume_connections)} volume "
                    f"outputs from single job {volume_from_job_uid}: {resolved_groups}"
                )
            else:
                if not volume_job_uids:
                    raise ValueError(
                        "heterogeneous_refinement: provide either volume_job_uids or "
                        "volume_from_job_uid."
                    )
                # Normalize a single UID / comma-separated string to a list (adapters
                # may forward a bare `volume_job_uid` string).
                if isinstance(volume_job_uids, str):
                    volume_job_uids = [v.strip() for v in volume_job_uids.split(",") if v.strip()]
                if num_classes is None:
                    num_classes = len(volume_job_uids)

                # If fewer volumes than requested classes were supplied (commonly a
                # SINGLE consensus volume + num_classes=K), repeat them to reach K so
                # the job actually runs K classes. CryoSPARC seeds K classes from K
                # identical volumes — this is the intended "one volume as K seeds"
                # behavior. Without this, K silently collapses to len(volume_job_uids).
                k = int(num_classes)
                if 0 < len(volume_job_uids) < k:
                    volume_job_uids = [volume_job_uids[i % len(volume_job_uids)] for i in range(k)]

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

                volume_connections = [(vol_job_uid, volume_slot) for vol_job_uid in volume_job_uids]
                num_classes = len(volume_connections)
                print(f"ℹ️  Heterogeneous refinement: connecting {len(volume_connections)} volumes (K={len(volume_connections)}) to 'volume' input group")
                print(f"ℹ️  All volumes from: {volume_job_uids[0]} (repeated {len(volume_connections)} times)")

            connections = {
                "particles": (particles_job_uid, particles_slot),
                "volume": volume_connections,
            }

            # Create the job
            # The number of classes (K) is automatically determined from the number of volume connections
            job = workspace.create_job(
                "hetero_refine",  # Heterogeneous refinement job type
                connections=connections,
                params=job_params
            )

            # Queue the job with lane auto-detection
            used_lane = self._queue_job_with_lane_fallback(
                job,
                lane=lane,
                hostname=hostname,
            )
            print(f"Queued heterogeneous refinement job: {job.uid}")

            job_uid = job.uid

            result = {
                "success": True,
                "job_uid": job_uid,
                "job_type": "heterogeneous_refinement",
                "message": f"Heterogeneous refinement job {job_uid} queued successfully",
                "num_classes": num_classes,
                "num_volumes": len(volume_connections),
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
            used_lane = self._queue_job_with_lane_fallback(
                job,
                lane=lane,
                hostname=hostname,
            )
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
            workspace_uid: Optional workspace UID containing the job (not used, kept for compatibility)
            
        Returns:
            Dictionary containing log content and analysis
        """
        try:
            # First, try to get the job directory using the CryoSPARC API
            log_file_path = None
            
            if project_uid:
                try:
                    # Use the API to get the actual job directory
                    job = self.cs.find_job(project_uid, job_uid)
                    job.refresh()
                    job_dir = str(job.dir())
                    log_file_path = os.path.join(job_dir, "job.log")
                    
                    # Verify the file exists
                    if not os.path.exists(log_file_path):
                        log_file_path = None
                except Exception as e:
                    # If API call fails, fall back to hardcoded paths
                    pass
            
            # Fallback: derive the job directory from the project's own directory
            # via the CryoSPARC API (no hardcoded host paths). The project knows its
            # real on-disk location, so job.log lives at <project_dir>/<job_uid>/job.log.
            if not log_file_path and project_uid:
                try:
                    project = self.cs.find_project(project_uid)
                    project_dir = str(project.dir())
                    candidate = os.path.join(project_dir, job_uid, "job.log")
                    if os.path.exists(candidate):
                        log_file_path = candidate
                except Exception:
                    pass
            
            if not log_file_path or not os.path.exists(log_file_path):
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
        Extract the actual error and warning lines from a job log.

        This does NOT classify errors into pre-defined categories or offer canned
        suggestions — it only reports what is genuinely present in the log, so the
        caller reasons about the real failure text (and decides what to do, e.g.
        consult the forum) rather than a guessed category.

        Args:
            log_content: Raw log content from the job

        Returns:
            Dictionary with: has_errors (bool), critical_errors (list of error
            lines from the log), warnings (list of warning lines), summary (str).
        """
        analysis: Dict[str, Any] = {
            "has_errors": False,
            "critical_errors": [],
            "warnings": [],
            "summary": "",
        }

        lines = log_content.split('\n')

        # Extract actual error lines from the log.
        for line in lines:
            lowered = line.lower()
            if 'error' in lowered and ('error:' in lowered or 'failed' in lowered):
                analysis["critical_errors"].append(line.strip())

        # Extract actual warning lines from the log.
        for line in lines:
            lowered = line.lower()
            if 'warning' in lowered or 'warn:' in lowered:
                analysis["warnings"].append(line.strip())

        analysis["has_errors"] = bool(analysis["critical_errors"])

        # Summary reflects only what was actually found in the log.
        if analysis["critical_errors"]:
            first = analysis["critical_errors"][0]
            analysis["summary"] = (
                f"Job log contains {len(analysis['critical_errors'])} error line(s); "
                f"first: {first}"
            )
        else:
            analysis["summary"] = "No error lines detected in log"

        return analysis

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
            used_lane = self._queue_job_with_lane_fallback(
                job_a,
                lane=lane,
                hostname=hostname,
            )
            second_lane = self._queue_job_with_lane_fallback(
                job_b,
                lane=used_lane if used_lane is not None else lane,
                hostname=hostname,
            )
            if used_lane is None:
                used_lane = second_lane
            
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
            used_lane = self._queue_job_with_lane_fallback(
                job,
                lane=lane,
                hostname=hostname,
            )

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

    # ==================================================================
    # SPA resolution-improvement + deep-picking tools (added 2026-06).
    # Each wrapper resolves input slots from the source jobs' real output
    # groups, merges friendly params + a raw `params` passthrough, creates
    # and queues the job, and returns {success, job_uid, job_type, message}.
    # ==================================================================

    def _create_and_queue_job(
        self,
        project_uid: str,
        workspace_uid: str,
        job_type: str,
        inputs: List[Any],
        job_params: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        passthrough_kwargs: Optional[Dict[str, Any]] = None,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        wait_for_completion: bool = False,
        timeout: int = 3600,
        check_interval: int = 30,
        result_extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Shared job runner for the new tools.

        `inputs` is a list of (connection_name, source_job_uid, slot_type),
        e.g. [("particles", "J10", "particle"), ("volume", "J20", "volume"),
        ("mask", "J30", "mask")]. A source_job_uid of None skips that connection
        (optional inputs like mask). Slot names are resolved from each source
        job's real output groups. Returns a result dict; never raises.
        """
        try:
            project = self.cs.find_project(project_uid)
            workspace = project.find_workspace(workspace_uid)

            connections: Dict[str, Any] = {}
            for conn_name, src_uid, slot_type in inputs:
                if not src_uid:
                    continue
                if slot_type == "particle":
                    slot = self._resolve_particles_slot(project, src_uid)
                else:
                    slot = self._resolve_output_slot(project, src_uid, slot_type)
                connections[conn_name] = (src_uid, slot)

            merged = self._merge_passthrough_params(
                dict(job_params or {}), params=params, kwargs=passthrough_kwargs
            )

            job = workspace.create_job(job_type, connections=connections, params=merged)
            used_lane = self._queue_job_with_lane_fallback(job, lane=lane, hostname=hostname)
            print(f"Queued {job_type} job: {job.uid}")

            result = {
                "success": True,
                "job_uid": job.uid,
                "job_type": job_type,
                "message": f"{job_type} job {job.uid} queued successfully",
                "lane": used_lane,
            }
            if result_extra:
                result.update(result_extra)
            if wait_for_completion:
                status_result = self.wait_for_job_completion(
                    project_uid=project_uid, job_uid=job.uid,
                    timeout=timeout, check_interval=check_interval)
                result.update(status_result)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "job_type": job_type,
                "message": f"Failed to queue {job_type} job: {str(e)}",
            }

    def ctf_refine_global(self, project_uid: str, workspace_uid: str,
                          particles_job_uid: str, volume_job_uid: str,
                          mask_job_uid: Optional[str] = None,
                          params: Optional[Dict[str, Any]] = None,
                          lane: Optional[str] = None, hostname: Optional[str] = None,
                          wait_for_completion: bool = False, timeout: int = 3600,
                          check_interval: int = 30, **kwargs) -> Dict[str, Any]:
        """Global CTF refinement (beam tilt / trefoil / higher-order aberrations).
        Inputs: particles, volume, optional mask. Tune via params keys crg_*."""
        return self._create_and_queue_job(
            project_uid, workspace_uid, "ctf_refine_global",
            [("particles", particles_job_uid, "particle"),
             ("volume", volume_job_uid, "volume"),
             ("mask", mask_job_uid, "mask")],
            params=params, passthrough_kwargs=kwargs, lane=lane, hostname=hostname,
            wait_for_completion=wait_for_completion, timeout=timeout, check_interval=check_interval)

    def ctf_refine_local(self, project_uid: str, workspace_uid: str,
                         particles_job_uid: str, volume_job_uid: str,
                         mask_job_uid: Optional[str] = None,
                         params: Optional[Dict[str, Any]] = None,
                         lane: Optional[str] = None, hostname: Optional[str] = None,
                         wait_for_completion: bool = False, timeout: int = 3600,
                         check_interval: int = 30, **kwargs) -> Dict[str, Any]:
        """Local (per-particle) defocus refinement. Inputs: particles, volume,
        optional mask. Tune via params keys crl_* (e.g. crl_df_range)."""
        return self._create_and_queue_job(
            project_uid, workspace_uid, "ctf_refine_local",
            [("particles", particles_job_uid, "particle"),
             ("volume", volume_job_uid, "volume"),
             ("mask", mask_job_uid, "mask")],
            params=params, passthrough_kwargs=kwargs, lane=lane, hostname=hostname,
            wait_for_completion=wait_for_completion, timeout=timeout, check_interval=check_interval)

    def local_refinement(self, project_uid: str, workspace_uid: str,
                         particles_job_uid: str, volume_job_uid: str,
                         mask_job_uid: Optional[str] = None,
                         params: Optional[Dict[str, Any]] = None,
                         lane: Optional[str] = None, hostname: Optional[str] = None,
                         wait_for_completion: bool = False, timeout: int = 3600,
                         check_interval: int = 30, **kwargs) -> Dict[str, Any]:
        """Local refinement of a masked region (new_local_refine). Inputs:
        particles, volume, mask (a focus mask strongly recommended)."""
        return self._create_and_queue_job(
            project_uid, workspace_uid, "new_local_refine",
            [("particles", particles_job_uid, "particle"),
             ("volume", volume_job_uid, "volume"),
             ("mask", mask_job_uid, "mask")],
            params=params, passthrough_kwargs=kwargs, lane=lane, hostname=hostname,
            wait_for_completion=wait_for_completion, timeout=timeout, check_interval=check_interval)

    def particle_subtract(self, project_uid: str, workspace_uid: str,
                          particles_job_uid: str, volume_job_uid: str,
                          mask_job_uid: Optional[str] = None,
                          params: Optional[Dict[str, Any]] = None,
                          lane: Optional[str] = None, hostname: Optional[str] = None,
                          wait_for_completion: bool = False, timeout: int = 3600,
                          check_interval: int = 30, **kwargs) -> Dict[str, Any]:
        """Subtract the masked region's signal from particles. Inputs: particles,
        volume, mask (mask defines the region to subtract)."""
        return self._create_and_queue_job(
            project_uid, workspace_uid, "particle_subtract",
            [("particles", particles_job_uid, "particle"),
             ("volume", volume_job_uid, "volume"),
             ("mask", mask_job_uid, "mask")],
            params=params, passthrough_kwargs=kwargs, lane=lane, hostname=hostname,
            wait_for_completion=wait_for_completion, timeout=timeout, check_interval=check_interval)

    def symmetry_expansion(self, project_uid: str, workspace_uid: str,
                           particles_job_uid: str, symmetry: Optional[str] = None,
                           params: Optional[Dict[str, Any]] = None,
                           lane: Optional[str] = None, hostname: Optional[str] = None,
                           wait_for_completion: bool = False, timeout: int = 3600,
                           check_interval: int = 30, **kwargs) -> Dict[str, Any]:
        """Symmetry-expand particles around a point group (sym_expand). Input:
        particles. `symmetry` -> sym_symmetry (e.g. C2, D7)."""
        job_params: Dict[str, Any] = {}
        if symmetry:
            job_params["sym_symmetry"] = symmetry
        return self._create_and_queue_job(
            project_uid, workspace_uid, "sym_expand",
            [("particles", particles_job_uid, "particle")],
            job_params=job_params, params=params, passthrough_kwargs=kwargs,
            lane=lane, hostname=hostname, wait_for_completion=wait_for_completion,
            timeout=timeout, check_interval=check_interval)

    def sharpen(self, project_uid: str, workspace_uid: str, volume_job_uid: str,
                mask_job_uid: Optional[str] = None, bfactor: Optional[float] = None,
                params: Optional[Dict[str, Any]] = None,
                lane: Optional[str] = None, hostname: Optional[str] = None,
                wait_for_completion: bool = False, timeout: int = 3600,
                check_interval: int = 30, **kwargs) -> Dict[str, Any]:
        """Sharpen a refined volume (B-factor / FSC weighting). Inputs: volume,
        optional mask. `bfactor` -> sharp_bfactor (negative sharpens)."""
        job_params: Dict[str, Any] = {}
        bf = self._coerce_float(bfactor)
        if bf is not None:
            job_params["sharp_bfactor"] = bf
        return self._create_and_queue_job(
            project_uid, workspace_uid, "sharpen",
            [("volume", volume_job_uid, "volume"), ("mask", mask_job_uid, "mask")],
            job_params=job_params, params=params, passthrough_kwargs=kwargs,
            lane=lane, hostname=hostname, wait_for_completion=wait_for_completion,
            timeout=timeout, check_interval=check_interval)

    def deepemhancer(self, project_uid: str, workspace_uid: str, volume_job_uid: str,
                     mask_job_uid: Optional[str] = None,
                     params: Optional[Dict[str, Any]] = None,
                     lane: Optional[str] = None, hostname: Optional[str] = None,
                     wait_for_completion: bool = False, timeout: int = 3600,
                     check_interval: int = 30, **kwargs) -> Dict[str, Any]:
        """DeepEMhancer deep-learning post-processing/sharpening. Inputs: volume,
        optional mask. Requires DeepEMhancer install on the worker."""
        return self._create_and_queue_job(
            project_uid, workspace_uid, "deepemhancer",
            [("volume", volume_job_uid, "volume"), ("mask", mask_job_uid, "mask")],
            params=params, passthrough_kwargs=kwargs, lane=lane, hostname=hostname,
            wait_for_completion=wait_for_completion, timeout=timeout, check_interval=check_interval)

    def local_resolution(self, project_uid: str, workspace_uid: str, volume_job_uid: str,
                         mask_job_uid: Optional[str] = None,
                         params: Optional[Dict[str, Any]] = None,
                         lane: Optional[str] = None, hostname: Optional[str] = None,
                         wait_for_completion: bool = False, timeout: int = 3600,
                         check_interval: int = 30, **kwargs) -> Dict[str, Any]:
        """Estimate a local resolution map from a refinement's half-maps. Inputs:
        volume (the refinement), optional mask."""
        return self._create_and_queue_job(
            project_uid, workspace_uid, "local_resolution",
            [("volume", volume_job_uid, "volume"), ("mask", mask_job_uid, "mask")],
            params=params, passthrough_kwargs=kwargs, lane=lane, hostname=hostname,
            wait_for_completion=wait_for_completion, timeout=timeout, check_interval=check_interval)

    def class_3d(self, project_uid: str, workspace_uid: str,
                 particles_job_uid: str, volume_job_uid: Optional[str] = None,
                 mask_job_uid: Optional[str] = None, focus_mask_job_uid: Optional[str] = None,
                 num_classes: Optional[int] = None,
                 params: Optional[Dict[str, Any]] = None,
                 lane: Optional[str] = None, hostname: Optional[str] = None,
                 wait_for_completion: bool = False, timeout: int = 3600,
                 check_interval: int = 30, **kwargs) -> Dict[str, Any]:
        """3D Classification (class_3D) on aligned particles. Inputs: particles,
        optional volume/mask/focus-mask. `num_classes` -> class3D_N_K."""
        job_params: Dict[str, Any] = {}
        if num_classes:
            job_params["class3D_N_K"] = int(num_classes)
        return self._create_and_queue_job(
            project_uid, workspace_uid, "class_3D",
            [("particles", particles_job_uid, "particle"),
             ("volume", volume_job_uid, "volume"),
             ("mask", mask_job_uid, "mask"),
             ("mask_focus", focus_mask_job_uid, "mask")],
            job_params=job_params, params=params, passthrough_kwargs=kwargs,
            lane=lane, hostname=hostname, wait_for_completion=wait_for_completion,
            timeout=timeout, check_interval=check_interval,
            result_extra={"num_classes": num_classes})

    def variability_3d(self, project_uid: str, workspace_uid: str,
                       particles_job_uid: str, mask_job_uid: Optional[str] = None,
                       num_modes: Optional[int] = None, symmetry: Optional[str] = None,
                       params: Optional[Dict[str, Any]] = None,
                       lane: Optional[str] = None, hostname: Optional[str] = None,
                       wait_for_completion: bool = False, timeout: int = 3600,
                       check_interval: int = 30, **kwargs) -> Dict[str, Any]:
        """3D Variability Analysis (var_3D) to probe continuous heterogeneity.
        Inputs: particles, mask. `num_modes` -> var_K, `symmetry` -> var_symmetry."""
        job_params: Dict[str, Any] = {}
        if num_modes:
            job_params["var_K"] = int(num_modes)
        if symmetry:
            job_params["var_symmetry"] = symmetry
        return self._create_and_queue_job(
            project_uid, workspace_uid, "var_3D",
            [("particles", particles_job_uid, "particle"),
             ("mask", mask_job_uid, "mask")],
            job_params=job_params, params=params, passthrough_kwargs=kwargs,
            lane=lane, hostname=hostname, wait_for_completion=wait_for_completion,
            timeout=timeout, check_interval=check_interval)

    def remove_duplicate_particles(self, project_uid: str, workspace_uid: str,
                                    particles_job_uid: str, micrographs_job_uid: Optional[str] = None,
                                    min_dist_A: Optional[float] = None,
                                    params: Optional[Dict[str, Any]] = None,
                                    lane: Optional[str] = None, hostname: Optional[str] = None,
                                    wait_for_completion: bool = False, timeout: int = 3600,
                                    check_interval: int = 30, **kwargs) -> Dict[str, Any]:
        """Remove duplicate particle picks within a minimum separation. Inputs:
        particles, optional micrographs. `min_dist_A` -> min_dist_A."""
        job_params: Dict[str, Any] = {}
        md = self._coerce_float(min_dist_A)
        if md is not None:
            job_params["min_dist_A"] = md
        return self._create_and_queue_job(
            project_uid, workspace_uid, "remove_duplicate_particles",
            [("particles", particles_job_uid, "particle"),
             ("micrographs", micrographs_job_uid, "exposure")],
            job_params=job_params, params=params, passthrough_kwargs=kwargs,
            lane=lane, hostname=hostname, wait_for_completion=wait_for_completion,
            timeout=timeout, check_interval=check_interval)

    def downsample_particles(self, project_uid: str, workspace_uid: str,
                             particles_job_uid: str, box_size_pix: Optional[int] = None,
                             bin_size_pix: Optional[int] = None,
                             params: Optional[Dict[str, Any]] = None,
                             lane: Optional[str] = None, hostname: Optional[str] = None,
                             wait_for_completion: bool = False, timeout: int = 3600,
                             check_interval: int = 30, **kwargs) -> Dict[str, Any]:
        """Downsample / Fourier-crop particles to a smaller box (faster early
        processing). Input: particles. `box_size_pix`, `bin_size_pix`."""
        job_params: Dict[str, Any] = {}
        if box_size_pix:
            job_params["box_size_pix"] = int(box_size_pix)
        if bin_size_pix:
            job_params["bin_size_pix"] = int(bin_size_pix)
        return self._create_and_queue_job(
            project_uid, workspace_uid, "downsample_particles",
            [("particles", particles_job_uid, "particle")],
            job_params=job_params, params=params, passthrough_kwargs=kwargs,
            lane=lane, hostname=hostname, wait_for_completion=wait_for_completion,
            timeout=timeout, check_interval=check_interval)

    def topaz_train(self, project_uid: str, workspace_uid: str,
                    micrographs_job_uid: str, particles_job_uid: str,
                    params: Optional[Dict[str, Any]] = None,
                    lane: Optional[str] = None, hostname: Optional[str] = None,
                    wait_for_completion: bool = False, timeout: int = 3600,
                    check_interval: int = 30, **kwargs) -> Dict[str, Any]:
        """Train a Topaz deep-picking model. Inputs: micrographs + a set of
        known-good particles to learn from. Requires Topaz on the worker."""
        return self._create_and_queue_job(
            project_uid, workspace_uid, "topaz_train",
            [("micrographs", micrographs_job_uid, "exposure"),
             ("particles", particles_job_uid, "particle")],
            params=params, passthrough_kwargs=kwargs, lane=lane, hostname=hostname,
            wait_for_completion=wait_for_completion, timeout=timeout, check_interval=check_interval)

    def topaz_extract(self, project_uid: str, workspace_uid: str,
                      model_job_uid: str, micrographs_job_uid: str,
                      params: Optional[Dict[str, Any]] = None,
                      lane: Optional[str] = None, hostname: Optional[str] = None,
                      wait_for_completion: bool = False, timeout: int = 3600,
                      check_interval: int = 30, **kwargs) -> Dict[str, Any]:
        """Pick particles with a trained/pretrained Topaz model. Inputs: model
        (from topaz_train) + micrographs. Alternative to blob/template picking."""
        return self._create_and_queue_job(
            project_uid, workspace_uid, "topaz_extract",
            [("model", model_job_uid, "ml_model"),
             ("micrographs", micrographs_job_uid, "exposure")],
            params=params, passthrough_kwargs=kwargs, lane=lane, hostname=hostname,
            wait_for_completion=wait_for_completion, timeout=timeout, check_interval=check_interval)

    def topaz_denoise(self, project_uid: str, workspace_uid: str,
                      micrographs_job_uid: str, denoise_model_job_uid: Optional[str] = None,
                      params: Optional[Dict[str, Any]] = None,
                      lane: Optional[str] = None, hostname: Optional[str] = None,
                      wait_for_completion: bool = False, timeout: int = 3600,
                      check_interval: int = 30, **kwargs) -> Dict[str, Any]:
        """Denoise micrographs with Topaz (improves picking on low-contrast data).
        Inputs: micrographs, optional pretrained denoise model."""
        return self._create_and_queue_job(
            project_uid, workspace_uid, "topaz_denoise",
            [("micrographs", micrographs_job_uid, "exposure"),
             ("denoise_model", denoise_model_job_uid, "ml_model")],
            params=params, passthrough_kwargs=kwargs, lane=lane, hostname=hostname,
            wait_for_completion=wait_for_completion, timeout=timeout, check_interval=check_interval)

    def class_2d_new(self, project_uid: str, workspace_uid: str,
                     particles_job_uid: str, num_classes: Optional[int] = None,
                     params: Optional[Dict[str, Any]] = None,
                     lane: Optional[str] = None, hostname: Optional[str] = None,
                     wait_for_completion: bool = False, timeout: int = 3600,
                     check_interval: int = 30, **kwargs) -> Dict[str, Any]:
        """2D Classification (new/faster engine, class_2D_new). Input: particles.
        `num_classes` -> class2D_K. Use as an alternative to legacy class_2D."""
        job_params: Dict[str, Any] = {}
        if num_classes:
            job_params["class2D_K"] = int(num_classes)
        return self._create_and_queue_job(
            project_uid, workspace_uid, "class_2D_new",
            [("particles", particles_job_uid, "particle")],
            job_params=job_params, params=params, passthrough_kwargs=kwargs,
            lane=lane, hostname=hostname, wait_for_completion=wait_for_completion,
            timeout=timeout, check_interval=check_interval,
            result_extra={"num_classes": num_classes})