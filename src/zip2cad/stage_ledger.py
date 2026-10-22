"""Only fingerprint-matched, hash-verified completed stages may be resumed."""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from .skp_exchange import ExchangeError, canonical_json, package_path, read_json, sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: dict) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def inventory(folder: Path) -> list[dict]:
    folder = folder.resolve()
    if not folder.is_dir():
        raise ExchangeError("INPUT_MISSING_FILE", f"Directory missing: {folder}")
    result = []
    for path in sorted(folder.rglob("*")):
        if not path.resolve().is_relative_to(folder):
            raise ExchangeError("INPUT_PATH", "Inventory symlink escapes its root")
        if path.is_file():
            result.append({"path": path.relative_to(folder).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return result


def code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "src" / "zip2cad").glob("*.py")) + sorted((root / "schemas").glob("*.json"))
    paths += [root / "tools" / "run_zipmap_block.py"]
    packages = {name: version(name) for name in ["numpy", "scipy", "trimesh", "pycollada", "Pillow", "opencv-python-headless", "jsonschema"]}
    return digest({"files": {p.relative_to(root).as_posix(): sha256_file(p) for p in paths},
                   "python": platform.python_version(), "packages": packages})


class StageLedger:
    def __init__(self, root: Path, identity: dict, resume: bool = False):
        # Compare the same JSON representation before and after persistence.
        # Dataclass tuple-valued configuration becomes a JSON list on disk.
        identity = json.loads(canonical_json(identity))
        self.root = root.resolve()
        self.path = self.root / "stage_ledger.json"
        if resume:
            self.data = read_json(self.path)
            if self.data.get("identity") != identity or self.data.get("identity_sha256") != digest(identity):
                raise ExchangeError("RESUME_IDENTITY", "Input/config/code fingerprints changed; use a new run directory")
            if self.data.get("format") != "zip2cad_stage_ledger_v1":
                raise ExchangeError("RESUME_FORMAT", "Not a managed workflow run")
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            self.data = {"format": "zip2cad_stage_ledger_v1", "identity": identity, "identity_sha256": digest(identity),
                         "created_at_utc": utc_now(), "stages": {}}
            self.save()

    def save(self) -> None:
        temporary = self.root / "stage_ledger.json.tmp"
        temporary.write_text(canonical_json(self.data), encoding="utf-8")
        temporary.replace(self.path)

    def run(self, name: str, action):
        stage = self.data["stages"].get(name)
        if stage and stage["status"] == "completed":
            folder = package_path(self.root, stage["folder"])
            if not stage.get("outputs") or inventory(folder) != stage["outputs"]:
                raise ExchangeError("RESUME_OUTPUT_HASH", f"Completed stage output changed: {name}")
            stage["last_verified_at_utc"] = utc_now()
            stage["reuse_count"] = stage.get("reuse_count", 0) + 1
            self.save()
            return folder, stage["result"], True
        # Never delete or overwrite a failed/running attempt; start a new one.
        folder = self.root / f"{name}_{uuid4().hex[:8]}"
        prior = [] if stage is None else stage.get("prior_attempts", []) + [{k: v for k, v in stage.items() if k != "prior_attempts"}]
        stage = {"status": "running", "folder": folder.name, "started_at_utc": utc_now(), "ended_at_utc": None,
                 "input_config_code_sha256": self.data["identity_sha256"], "prior_attempts": prior, "reuse_count": 0}
        self.data["stages"][name] = stage
        self.save()
        try:
            result = action(folder)
            outputs = inventory(folder)
            if not outputs:
                raise ExchangeError("STAGE_EMPTY", f"Stage produced no artifacts: {name}")
            stage.update(status="completed", ended_at_utc=utc_now(), result=result, outputs=outputs)
            self.save()
            return folder, result, False
        except Exception as error:
            stage.update(status="failed", ended_at_utc=utc_now(), error=str(error))
            self.save()
            raise
