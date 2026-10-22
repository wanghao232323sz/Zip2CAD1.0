"""Strict, read-only validation of version 1 SketchUp exchange packages.

Transforms are row-major 4x4 matrices acting on column vectors. Translation
and geometry are expressed in metres; identity is namespace + persistent path.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from zip2cad.dae_scene import read_collada_asset
from zip2cad.exchange_geometry import load_exchange_geometry


class ExchangeError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    def reject_constant(value: str) -> None:
        raise ExchangeError("INPUT_NONFINITE", f"{path.name}: {value}")

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ExchangeError("INPUT_DUPLICATE_KEY", f"{path.name}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"),
                           parse_constant=reject_constant, object_pairs_hook=unique_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExchangeError("INPUT_JSON", f"{path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ExchangeError("INPUT_SCHEMA", f"{path.name}: expected an object")

    def finite_numbers(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ExchangeError("INPUT_NONFINITE", f"{path.name}: non-finite numeric value")
        if isinstance(item, dict):
            for child in item.values():
                finite_numbers(child)
        elif isinstance(item, list):
            for child in item:
                finite_numbers(child)

    finite_numbers(value)
    return value


def validate_document(value: dict, schema_name: str) -> None:
    schema_path = Path(__file__).resolve().parents[2] / "schemas" / f"{schema_name}.schema.json"
    if not schema_path.is_file():
        schema_path = Path(sys.prefix) / "share" / "zip2cad" / "schemas" / f"{schema_name}.schema.json"
    schema = read_json(schema_path)
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda e: str(list(e.path)))
    if errors:
        error = errors[0]
        location = ".".join(str(p) for p in error.path) or "$"
        raise ExchangeError("INPUT_SCHEMA", f"{schema_name}.{location}: {error.message}")


def package_path(root: Path, relative: str) -> Path:
    # Reject Windows paths, traversal, alternate data streams and symlink escapes.
    path = PurePosixPath(relative)
    if not relative or "\\" in relative or ":" in relative or path.is_absolute() or ".." in path.parts:
        raise ExchangeError("INPUT_PATH", f"Unsafe package path: {relative}")
    target = (root / path).resolve()
    if target == root or not target.is_relative_to(root):
        raise ExchangeError("INPUT_PATH", f"Path escapes package: {relative}")
    return target


def stable_entity_id(namespace: str, pid_path: str) -> str:
    return f"skp:{namespace}:{pid_path}"


@dataclass(frozen=True)
class ExchangeEntity:
    entity_id: str
    persistent_id_path: str
    entity_type: str
    parent_entity_id: str | None
    editable: bool
    local_transform: tuple[tuple[float, ...], ...]
    world_transform: tuple[tuple[float, ...], ...]
    bbox_world_m: tuple[tuple[float, ...], tuple[float, ...]]
    vertex_refs: tuple[int, ...]
    face_refs: tuple[int, ...]
    locked_reason: str | None = None
    tag: str = ""


@dataclass(frozen=True)
class ExchangePackage:
    root: Path
    project_id: str
    model_namespace: str
    source_model_sha256: str
    geometry_path: Path
    entities: tuple[ExchangeEntity, ...]


def load_exchange(directory: str | Path) -> ExchangePackage:
    root = Path(directory).resolve()
    manifest = read_json(root / "manifest.json")
    validate_document(manifest, "exchange_manifest")
    files = {}
    for entry in manifest["files"]:
        relative = entry["path"]
        if relative in files or relative == "manifest.json":
            raise ExchangeError("INPUT_MANIFEST", f"Duplicate or self-referencing manifest entry: {relative}")
        path = package_path(root, relative)
        if not path.is_file():
            raise ExchangeError("INPUT_MISSING_FILE", relative)
        if path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]:
            raise ExchangeError("INPUT_HASH_MISMATCH", relative)
        files[relative] = path
    for required in ("project.json", "entities.json", "scene.dae", "materials.json", "preview.png"):
        if required not in files:
            raise ExchangeError("INPUT_MANIFEST", f"Missing manifest entry: {required}")
    project = read_json(files["project.json"])
    document = read_json(files["entities.json"])
    validate_document(project, "project")
    validate_document(document, "entities")
    if document["model_namespace"] != project["model_namespace"]:
        raise ExchangeError("INPUT_IDENTITY", "Project and entities namespaces differ")
    records = document["entities"]
    by_id = {}
    for record in records:
        entity_id = record["entity_id"]
        if entity_id in by_id:
            raise ExchangeError("INPUT_DUPLICATE_ID", entity_id)
        if entity_id != stable_entity_id(project["model_namespace"], record["persistent_id_path"]):
            raise ExchangeError("INPUT_IDENTITY", entity_id)
        if int(record["persistent_id_path"].split(".")[-1]) != record["persistent_id"]:
            raise ExchangeError("INPUT_IDENTITY", f"Leaf PID mismatch: {entity_id}")
        for key in ("local_transform", "world_transform"):
            matrix = np.asarray(record[key], dtype=float)
            if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
                raise ExchangeError("INPUT_TRANSFORM", f"Invalid affine transform: {entity_id}.{key}")
            if abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
                raise ExchangeError("INPUT_TRANSFORM", f"Singular transform: {entity_id}.{key}")
        for key in ("bbox_local_m", "bbox_world_m"):
            bounds = np.asarray([record[key]["min"], record[key]["max"]], dtype=float)
            if not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1]):
                raise ExchangeError("INPUT_BOUNDS", f"Invalid bounds: {entity_id}.{key}")
        if record["editable"] and record["locked_reason"] is not None:
            raise ExchangeError("INPUT_EDITABLE", f"Editable entity has lock reason: {entity_id}")
        by_id[entity_id] = record
    roots = [record for record in records if record["parent_entity_id"] is None]
    if len(roots) != 1 or roots[0]["entity_type"] not in ("group", "component_instance"):
        raise ExchangeError("INPUT_HIERARCHY", "Expected exactly one group/component root")
    for record in records:
        parent_id = record["parent_entity_id"]
        if parent_id is None:
            continue
        parent = by_id.get(parent_id)
        if parent is None or parent["entity_type"] not in ("group", "component_instance"):
            raise ExchangeError("INPUT_HIERARCHY", f"Invalid parent: {record['entity_id']}")
        expected_path = f"{parent['persistent_id_path']}.{record['persistent_id']}"
        if record["persistent_id_path"] != expected_path:
            raise ExchangeError("INPUT_HIERARCHY", f"Invalid instance path: {record['entity_id']}")
        expected = np.asarray(parent["world_transform"]) @ np.asarray(record["local_transform"])
        if not np.allclose(expected, record["world_transform"], rtol=1e-9, atol=1e-8):
            raise ExchangeError("INPUT_TRANSFORM", f"Parent/local/world mismatch: {record['entity_id']}")
        if record["editable"] and not parent["editable"]:
            raise ExchangeError("INPUT_EDITABLE", f"Locked parent: {record['entity_id']}")
    # v1 scene.dae is the flat, metre/Z-up triangle mesh written with the ID map.
    # Never apply face_refs to a DAE exported by another tool without verifying it.
    try:
        _name, unit_meter, up_axis = read_collada_asset(files["scene.dae"])
        if unit_meter != 1.0 or up_axis != "Z_UP":
            raise ExchangeError("INPUT_GEOMETRY_UNIT", "scene.dae must use metres and Z_UP")
        scene = load_exchange_geometry(files["scene.dae"])
    except ExchangeError:
        raise
    except Exception as error:
        raise ExchangeError("INPUT_GEOMETRY", f"Unable to read scene.dae: {error}") from error
    if len(scene.instance_names) != 1 or not np.isfinite(scene.vertices_world_m).all():
        raise ExchangeError("INPUT_GEOMETRY", "Expected one finite flat triangle mesh")
    owned_faces = []
    for record in records:
        vertex_refs = record["vertex_refs"]
        face_refs = record["face_refs"]
        if any(index >= len(scene.vertices_world_m) for index in vertex_refs):
            raise ExchangeError("INPUT_GEOMETRY_MAP", f"Vertex index out of range: {record['entity_id']}")
        if any(index >= len(scene.faces) for index in face_refs):
            raise ExchangeError("INPUT_GEOMETRY_MAP", f"Face index out of range: {record['entity_id']}")
        if record["entity_type"] == "face":
            if not face_refs or not vertex_refs:
                raise ExchangeError("INPUT_GEOMETRY_MAP", f"Empty face mapping: {record['entity_id']}")
            used = set(scene.faces[face_refs].ravel().tolist())
            if not used.issubset(set(vertex_refs)):
                raise ExchangeError("INPUT_GEOMETRY_MAP", f"Triangle references another entity: {record['entity_id']}")
            owned_faces.extend(face_refs)
    if sorted(owned_faces) != list(range(len(scene.faces))):
        raise ExchangeError("INPUT_GEOMETRY_MAP", "Every DAE triangle must have exactly one face owner")
    entities = tuple(ExchangeEntity(
        record["entity_id"], record["persistent_id_path"], record["entity_type"],
        record["parent_entity_id"], record["editable"],
        tuple(tuple(row) for row in record["local_transform"]),
        tuple(tuple(row) for row in record["world_transform"]),
        (tuple(record["bbox_world_m"]["min"]), tuple(record["bbox_world_m"]["max"])),
        tuple(record["vertex_refs"]), tuple(record["face_refs"]), record["locked_reason"], record["tag"])
        for record in sorted(records, key=lambda r: r["entity_id"]))
    return ExchangePackage(root, project["project_id"], project["model_namespace"],
                           project["source_model_sha256"], files["scene.dae"], entities)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Validate a SketchUp exchange package without modifying it")
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    try:
        package = load_exchange(args.package)
    except ExchangeError as error:
        parser.exit(1, f"{error}\n")
    print(canonical_json({"project_id": package.project_id, "entity_count": len(package.entities),
                          "status": "validated"}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
