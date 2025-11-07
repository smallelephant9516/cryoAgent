"""CryoSift tooling for CryoAgent."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class CryoSiftPaths:
    """Configuration of key CryoSift paths."""

    cryosift_root: Path = Path("/home/daoyi/tools/cryosift")
    evaluator_script: Path = Path(
        "/home/daoyi/tools/cryosift/2dclass_evaluator/CNNTraining/output_class_list.py"
    )
    default_weights: Path = Path(
        "/home/daoyi/tools/cryosift/2dclass_evaluator/CNNTraining/final_model/final_model_cont.pth"
    )

    def resolve(self) -> "CryoSiftPaths":
        """Resolve all configured paths to absolute filesystem locations."""

        return CryoSiftPaths(
            cryosift_root=self.cryosift_root.expanduser().resolve(),
            evaluator_script=self.evaluator_script.expanduser().resolve(),
            default_weights=self.default_weights.expanduser().resolve(),
        )


class CryoSiftTools:
    """Convenience wrapper for running CryoSift utilities."""

    def __init__(
        self,
        paths: Optional[CryoSiftPaths] = None,
        python_executable: str = "python",
        *,
        conda_env: Optional[str] = "magellon2DAssess",
        conda_executable: str = "conda",
    ) -> None:
        """Initialise the tool wrapper.

        Args:
            paths: Optional path configuration. Falls back to default locations
                under ``/home/daoyi/tools/cryosift``.
            python_executable: Python interpreter to use when launching
                CryoSift scripts.
        """

        self._paths = (paths or CryoSiftPaths()).resolve()
        self._python = python_executable
        self._conda_env = conda_env
        self._conda_executable = conda_executable

        if not self._paths.evaluator_script.exists():
            raise FileNotFoundError(
                f"CryoSift evaluator script not found: {self._paths.evaluator_script}"
            )

    @staticmethod
    def _normalise_output_dir(output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def _normalise_weights_path(
        weights_path: Optional[Path],
        classification_dir: Path,
        fallback_weights: Path,
    ) -> Path:
        if weights_path is None:
            return fallback_weights

        if weights_path.is_absolute():
            return weights_path

        # Treat relative paths as relative to the classification directory
        candidate = (classification_dir / weights_path).resolve()
        if candidate.exists():
            return candidate

        # Fall back to user-provided (relative) path resolved from current directory
        return weights_path.resolve()

    def run_classification_evaluator(
        self,
        classification_dir: Path | str,
        output_dir: Path | str,
        *,
        weights_path: Optional[Path | str] = None,
        threshold: int = 3,
        extra_args: Optional[Sequence[str]] = None,
        capture_output: bool = True,
        text: bool = True,
        timeout: Optional[float] = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
        """Run the CryoSift 2D-class evaluator.

        Args:
            classification_dir: CryoSPARC 2D classification result directory
                used as input (``-i``).
            output_dir: Destination directory for CryoSift outputs (``-o``).
            weights_path: Optional path for classifier weights (``-w``).
                If relative, it is interpreted relative to ``classification_dir``.
                Defaults to the pre-trained weights bundled with CryoSift.
            threshold: Confidence threshold (``-t``) passed to the evaluator.
            extra_args: Additional CLI arguments appended verbatim.
            capture_output: Whether to capture stdout/stderr from the process.
            text: Return captured output as text if ``capture_output`` is True.
            timeout: Optional timeout in seconds for subprocess execution.
            check: If True, raise ``CalledProcessError`` when the command fails.

        Returns:
            The ``subprocess.CompletedProcess`` instance for the invocation.
        """

        classification_path = Path(classification_dir).expanduser().resolve()
        if not classification_path.exists():
            raise FileNotFoundError(
                f"Input classification directory does not exist: {classification_path}"
            )

        output_path = self._normalise_output_dir(Path(output_dir).expanduser().resolve())

        weights_candidate = (
            Path(weights_path).expanduser()
            if weights_path is not None
            else None
        )
        weights_final = self._normalise_weights_path(
            weights_candidate,
            classification_path,
            self._paths.default_weights,
        )

        if not weights_final.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_final}")

        cmd: list[str] = []

        if self._conda_env:
            cmd.extend(
                [
                    self._conda_executable,
                    "run",
                    "-n",
                    self._conda_env,
                    self._python,
                ]
            )
        else:
            cmd.append(self._python)

        cmd.extend(
            [
                str(self._paths.evaluator_script),
            "-i",
            str(classification_path),
            "-o",
            str(output_path),
            "-w",
            str(weights_final),
            "-t",
            str(threshold),
            ]
        )

        if extra_args:
            cmd.extend(str(arg) for arg in _flatten(extra_args))

        return subprocess.run(  # noqa: S603
            cmd,
            check=check,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
        )

    def evaluate_and_get_selected_classes(
        self,
        classification_dir: Path | str,
        output_dir: Path | str,
        *,
        weights_path: Optional[Path | str] = None,
        threshold: float = 3.0,
        extra_args: Optional[Sequence[str]] = None,
    ) -> Tuple[List[int], Dict[int, float], Path]:
        """Run CryoSift and collect selected class indices.

        Returns the list of selected class indices, a mapping of indices to
        their CryoSift scores, and the directory containing CryoSift outputs.
        """

        classification_path = Path(classification_dir)
        output_path = Path(output_dir)

        try:
            self.run_classification_evaluator(
                classification_path,
                output_path,
                weights_path=weights_path,
                threshold=threshold,
                extra_args=extra_args,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            # CryoSift may exit with non-zero status after generating outputs (e.g., score.txt bug).
            print(f"⚠️ CryoSift exited with code {exc.returncode}: {exc}")
            if not output_path.exists():
                raise

        score_file = self._locate_score_file(output_path, threshold)
        if score_file is None:
            model_scores = self._locate_model_scores(output_path)
            if not model_scores:
                raise FileNotFoundError(
                    f"CryoSift score file not found in {output_path}. Expected 'score.txt', 'score_t*.star', or '*_model.star'."
                )
            indices = [idx for idx, score in model_scores.items() if score is not None and score < threshold]
            scores = {idx: model_scores[idx] for idx in indices}
            self._write_score_txt(output_path, indices, scores)
        else:
            indices, scores = self._parse_score_file(score_file)
        return indices, scores, output_path

    @staticmethod
    def _locate_score_file(directory: Path, threshold: float) -> Optional[Path]:
        candidates = [directory / "score.txt"]
        pattern = re.compile(r"score[_-]?t?\s*" + re.escape(str(threshold)))

        for path in sorted(directory.glob("score*")):
            if path.is_file() and path.suffix.lower() in {".star", ".txt"}:
                if path.name == "score.txt" or pattern.search(path.stem.lower()):
                    candidates.append(path)

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _parse_score_file(score_file: Path) -> Tuple[List[int], Dict[int, float]]:
        indices: List[int] = []
        scores: Dict[int, float] = {}

        with score_file.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        in_data_section = False
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("loop_"):
                in_data_section = True
                continue
            if line.startswith("_"):
                continue

            parts = line.split()
            if not parts:
                continue

            try:
                index = int(parts[0])
            except ValueError:
                if not in_data_section:
                    continue
                # STAR files often include references like '00001@file'. Extract leading digits.
                match = re.match(r"(\d+)", parts[0])
                if not match:
                    continue
                index = int(match.group(1))
                if "@" in parts[0]:
                    index -= 1  # STAR references are typically 1-based
            else:
                if "@" in parts[0]:
                    index = max(0, index - 1)

            score: Optional[float] = None
            if len(parts) > 1:
                try:
                    score = float(parts[1])
                except ValueError:
                    score = None

            indices.append(index)
            if score is not None:
                scores[index] = score

        return indices, scores

    @staticmethod
    def _locate_model_scores(directory: Path) -> Dict[int, float]:
        for candidate in sorted(directory.glob("*_model.star")):
            if candidate.is_file():
                _, scores = CryoSiftTools._parse_score_file(candidate)
                return scores
        return {}

    @staticmethod
    def _write_score_txt(directory: Path, indices: List[int], scores: Dict[int, float]) -> None:
        if not indices:
            return
        score_path = directory / "score.txt"
        with score_path.open("w", encoding="utf-8") as f:
            for idx in indices:
                score = scores.get(idx)
                if score is None:
                    f.write(f"{idx}\n")
                else:
                    f.write(f"{idx} {score:.3f}\n")


def _flatten(args: Iterable[str | Iterable[str]]) -> Iterable[str]:
    """Flatten nested iterables of CLI arguments."""

    for arg in args:
        if isinstance(arg, (list, tuple)):
            yield from _flatten(arg)
        else:
            yield str(arg)

