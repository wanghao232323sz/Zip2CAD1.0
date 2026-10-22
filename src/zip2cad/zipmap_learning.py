"""ZipMap-only input audit and source-preserving learning data preparation.

This module does not load Gaussian models, historical completion runs, or any
neural training dependency. Source-frame checks of an already fused point cloud
are internal checks, never independent image reconstruction tests.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from shapely import MultiPoint, contains_xy

from .skp_exchange import canonical_json, sha256_file as _sha256_file


def sha256_file(path):
    return _sha256_file(Path(path))

PATH_FIELDS = ("source_model", "source_video", "frame_manifest", "fusion_archive",
               "building_cloud", "fusion_report", "provenance")
CONFIG_FIELDS = set(PATH_FIELDS) | {
    "schema", "project_root", "project_id", "pointcloud_source", "coordinate_system",
    "scope_pid_path", "trusted_existing_exterior", "expected_source_sha256", "camera_image_size",
    "expected_cloud_sha256", "crop", "normals", "baseline", "internal_check_stride",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(canonical_json(value), encoding="utf-8")


def inside(root, value):
    path = (Path(root) / value).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Project data and outputs must remain inside project_root: " + str(path))
    return path


def load_project(path):
    path = Path(path).resolve()
    c = read_json(path)
    if set(c) - CONFIG_FIELDS:
        raise ValueError("Unsupported input/config fields: " + str(sorted(set(c) - CONFIG_FIELDS)))
    if c.get("schema") != "zip2cad_zipmap_learning_project_v1" or c.get("pointcloud_source") != "ZipMap":
        raise ValueError("An explicit ZipMap-only learning configuration is required")
    if c.get("coordinate_system") != "CAD_metres" or c.get("trusted_existing_exterior") is not True:
        raise ValueError("Explicit CAD metre coordinates and trusted exterior premise required")
    if len(c.get("camera_image_size", [])) != 2 or any(type(x) is not int or x <= 0 for x in c["camera_image_size"]):
        raise ValueError("Explicit positive prediction image width and height required")
    scope = c.get("scope_pid_path")
    if not isinstance(scope, list) or not scope or any(type(x) is not int or x <= 0 for x in scope):
        raise ValueError("A nonempty source CAD instance PID path is required")
    # Only declared fields are dereferenced. Paths inside saved historical
    # provenance are metadata and are deliberately not followed.
    root = (path.parent / c["project_root"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Configuration must belong to its project_root")
    resources = {k: inside(root, c[k]) for k in PATH_FIELDS}
    for k, resource in resources.items():
        if not resource.is_file():
            raise ValueError("Missing " + k + ": " + str(resource))
    for name, allowed in {
        "crop": {"xy_margin_m", "lower_margin_m", "upper_margin_m"},
        "normals": {"neighbors", "radius_m", "minimum_neighbors", "maximum_curvature", "minimum_tangent_ratio"},
        "baseline": {"minimum_height_above_support_m", "maximum_height_above_support_m", "voxel_m", "plane_tolerance_m", "minimum_area_m2", "minimum_change_m", "minimum_training_views", "minimum_heldout_fraction", "maximum_components", "thickness_prior_m", "minimum_reliable_normal_fraction"},
    }.items():
        if not isinstance(c.get(name), dict) or set(c[name]) - allowed:
            raise ValueError("Unknown settings in " + name)
        if any(isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in c[name].values()):
            raise ValueError("Positive finite numeric settings required in " + name)
    stride = c.get("internal_check_stride", 5)
    if type(stride) is not int or stride < 2:
        raise ValueError("Internal check stride must be an integer >= 2")
    return c, root, resources


def fresh_output(root, output, inputs=()):
    output = inside(root, output)
    if output.exists():
        raise ValueError("Fresh output directory required: " + str(output))
    for resource in inputs:
        resource = Path(resource).resolve()
        if resource == output or resource.is_relative_to(output):
            raise ValueError("Output must not contain any input")
    output.mkdir(parents=True)
    return output


def validate_fusion(data):
    required = {"points", "colours", "votes", "frame_refs", "camera_refs", "camera_K", "camera_R", "camera_C"}
    if required - set(data):
        raise ValueError("Missing fusion fields: " + str(required - set(data)))
    p, rgb, refs, votes = (data[k] for k in ("points", "colours", "frame_refs", "votes"))
    n = len(p)
    if p.shape != (n, 3) or n == 0 or not np.isfinite(p).all():
        raise ValueError("Finite nonempty Nx3 point coordinates required")
    if rgb.shape != p.shape or rgb.dtype != np.uint8 or refs.shape != (n,) or votes.shape != (n,):
        raise ValueError("Fusion point attributes have inconsistent shape or colour dtype")
    if not np.issubdtype(refs.dtype, np.integer) or not np.issubdtype(votes.dtype, np.integer) or np.any(votes < 0):
        raise ValueError("Integer source references and nonnegative support counts required")
    cr = data["camera_refs"]
    if cr.ndim != 1 or not np.issubdtype(cr.dtype, np.integer) or len(np.unique(cr)) != len(cr):
        raise ValueError("Unique integer camera references required")
    if not np.isin(refs, cr).all():
        raise ValueError("Point source frame has no corresponding camera")
    for k, shape in (("camera_K", (len(cr), 3, 3)), ("camera_R", (len(cr), 3, 3)), ("camera_C", (len(cr), 3))):
        if data[k].shape != shape or not np.isfinite(data[k]).all():
            raise ValueError("Invalid camera array " + k)
    rotations = data["camera_R"]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-4) or np.any(np.linalg.det(rotations) < 0.999):
        raise ValueError("Proper world-to-camera rotations required")
    if np.any(np.linalg.det(data["camera_K"]) <= 0):
        raise ValueError("Nondegenerate positive intrinsics required")
    return {"scene_points": n, "source_frame_count": int(len(np.unique(refs))), "camera_count": len(cr)}


def load_fusion(path):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    validate_fusion(data)
    return data


def audit_inputs(config_path, output):
    c, root, paths = load_project(config_path)
    # Validate the saved data source before SDK export or any output mutation.
    report, provenance = read_json(paths["fusion_report"]), read_json(paths["provenance"])
    if report.get("Gaussian_points_used") is not False or report.get("CAD_surface_points_used") is not False:
        raise ValueError("Fusion source must explicitly exclude Gaussian/CAD point substitution")
    if provenance.get("Gaussian_geometry_used") is not False or provenance.get("CAD_surface_substitution_used") is not False:
        raise ValueError("Unverified point-cloud geometry source")
    hashes = {k: sha256_file(p) for k, p in paths.items()}
    for field, key in (("expected_source_sha256", "source_model"), ("expected_cloud_sha256", "building_cloud")):
        if c.get(field) and c[field] != hashes[key]:
            raise ValueError("Input hash changed: " + key)
    data = load_fusion(paths["fusion_archive"])
    summary = validate_fusion(data)
    if summary["scene_points"] != report["fused_scene_points"] or summary["camera_count"] != report["all_source_frame_count"]:
        raise ValueError("Fusion archive/report counts disagree")
    manifest = read_json(paths["frame_manifest"])
    accepted = {row["accepted_index"] - 1: row for row in manifest["frames"] if row["status"] == "accepted"}
    if not set(map(int, data["camera_refs"])) <= set(accepted):
        raise ValueError("Fusion camera references cannot be bound to source video frames")
    if Path(manifest["source_video"]["path"]).resolve() != paths["source_video"] or manifest["source_video"]["size_bytes"] != paths["source_video"].stat().st_size:
        raise ValueError("Source video differs from the accepted frame manifest")
    output = fresh_output(root, output, paths.values())
    from .headless_cli import export_cad
    context = export_cad(paths["source_model"], output / "source_cad.npz", c["scope_pid_path"], c["project_id"])
    resolved = dict(c, project_root=str(root), resources={k: {"path": str(p), "sha256": hashes[k]} for k, p in paths.items()},
                    config_path=str(Path(config_path).resolve()), config_sha256=sha256_file(config_path), CAD_context=context,
                    source_cad=str(output / "source_cad.npz"), source_cad_sha256=sha256_file(output / "source_cad.npz"))
    save_json(output / "resolved_inputs.json", resolved)
    audit = {"schema": "zip2cad_zipmap_input_audit_v1", "status": "passed", **summary,
             "pointcloud_source": "ZipMap", "Gaussian_resources_required": False,
             "historical_paths_in_provenance_dereferenced": False,
             "source_video_hash": hashes["source_video"], "input_hashes": hashes,
             "coordinate_system": "CAD_metres", "new_transform_applied": False,
             "CAD_context": context, "independent_survey_precision_certified": False,
             "accepted_manifest_frame_count": len(accepted),
             "source_files_unchanged": all(sha256_file(paths[k]) == hashes[k] for k in ("source_model", "fusion_archive", "building_cloud"))}
    save_json(output / "input_audit.json", audit)
    return audit


def checked_audit(path):
    path = Path(path).resolve()
    a = read_json(path / "resolved_inputs.json")
    root = Path(a["project_root"]).resolve()
    inside(root, path)
    if read_json(path / "input_audit.json")["status"] != "passed":
        raise ValueError("Successful Step 1 audit required")
    load_project(a["config_path"])
    if sha256_file(a["config_path"]) != a["config_sha256"]:
        raise ValueError("Project configuration changed after audit")
    for key in ("fusion_archive", "source_model", "building_cloud", "fusion_report", "provenance"):
        resource = a["resources"][key]
        inside(root, resource["path"])
        if sha256_file(resource["path"]) != resource["sha256"]:
            raise ValueError("Audited resource changed: " + key)
    inside(root, a["source_cad"])
    if sha256_file(a["source_cad"]) != a["source_cad_sha256"]:
        raise ValueError("Audited CAD export changed")
    return a, root


def scoped_vertices(cad, scope, project_id):
    prefix = "skp:" + project_id + ":" + ".".join(map(str, scope)) + "."
    selected = np.array([str(x).startswith(prefix) for x in cad["entity_ids"]])
    faces = cad["faces"][selected[cad["face_owner_indices"]]]
    if len(faces) < 4:
        raise ValueError("Target CAD instance has too few faces")
    return cad["vertices_world_m"][np.unique(faces)]


def crop_mask(points, vertices, attachment_z, settings):
    polygon = MultiPoint(vertices[:, :2]).convex_hull.buffer(settings["xy_margin_m"])
    low = float(vertices[:, 2].min() - settings["lower_margin_m"])
    high = float(max(vertices[:, 2].max(), attachment_z) + settings["upper_margin_m"])
    return contains_xy(polygon, points[:, 0], points[:, 1]) & (points[:, 2] > low) & (points[:, 2] < high), [low, high]


def estimate_normals(points, source_centres, settings, batch_size=50000):
    points = np.asarray(points, dtype=np.float64)
    tree = cKDTree(points)
    result = np.zeros(points.shape, np.float32)
    valid = np.zeros(len(points), bool)
    curvature = np.ones(len(points), np.float32)
    k = min(int(settings["neighbors"]), len(points))
    if k < 3:
        return result, valid, curvature
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        distance, indices = tree.query(points[start:stop], k=k, workers=4)
        near = distance <= settings["radius_m"]
        count = near.sum(axis=1)
        neighbours = points[indices]
        centre = (neighbours * near[..., None]).sum(axis=1) / np.maximum(count[:, None], 1)
        delta = (neighbours - centre[:, None]) * near[..., None]
        covariance = np.einsum("nki,nkj->nij", delta, delta) / np.maximum(count[:, None, None], 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0)
        curve = eigenvalues[:, 0] / np.maximum(eigenvalues.sum(axis=1), 1e-12)
        good = (count >= settings["minimum_neighbors"]) & (curve <= settings["maximum_curvature"]) & (eigenvalues[:, 1] / np.maximum(eigenvalues[:, 2], 1e-12) >= settings["minimum_tangent_ratio"])
        normals = eigenvectors[:, :, 0]
        orientation = np.einsum("ij,ij->i", normals, source_centres[start:stop] - points[start:stop])
        normals[orientation < 0] *= -1
        normals[~good] = 0
        result[start:stop], valid[start:stop], curvature[start:stop] = normals, good, curve
        print(f"normal estimation {stop}/{len(points)}", flush=True)
    return result, valid, curvature


def adapt_fusion(audit_path, output):
    a, root = checked_audit(audit_path)
    data = load_fusion(a["resources"]["fusion_archive"]["path"])
    with np.load(a["source_cad"], allow_pickle=False) as z:
        vertices = scoped_vertices(z, a["scope_pid_path"], a["project_id"])
    mask, bounds = crop_mask(data["points"], vertices, a["CAD_context"]["CAD_attachment_z_m"], a["crop"])
    indices = np.flatnonzero(mask)
    p, rgb, refs, votes = (data[k][indices] for k in ("points", "colours", "frame_refs", "votes"))
    # Reproduce the frozen building PLY crop, including order and colours.
    ply = trimesh.load(a["resources"]["building_cloud"]["path"], process=False)
    if not np.array_equal(p, np.asarray(ply.vertices, np.float32)) or not np.array_equal(rgb, np.asarray(ply.colors[:, :3])):
        raise ValueError("Rebuilt CAD crop does not exactly match the saved building PLY")
    positions = {int(ref): i for i, ref in enumerate(data["camera_refs"])}
    centres = data["camera_C"][[positions[int(ref)] for ref in refs]]
    normals, valid, curvature = estimate_normals(p, centres, a["normals"])
    origin = (p.min(axis=0).astype(float) + p.max(axis=0).astype(float)) / 2
    scale = float(np.ptp(p.astype(float), axis=0).max())
    local = ((p.astype(float) - origin) / scale).astype(np.float32)
    roundtrip = float(np.max(abs(local.astype(float) * scale + origin - p)))
    if roundtrip > 1e-4:
        raise ValueError("Local-coordinate roundtrip exceeds 0.1 mm numerical tolerance")
    output = fresh_output(root, output, [Path(audit_path), *[v["path"] for v in a["resources"].values()]])
    np.savez_compressed(output / "observed.npz", points_CAD_m=p, points_local=local, rgb=rgb,
                        normal=normals, normal_valid=valid, normal_curvature=curvature,
                        frame_refs=refs, support_count=votes)
    np.savez_compressed(output / "source_index.npz", scene_indices=indices)
    camera = {k: data[k].tolist() for k in ("camera_refs", "camera_K", "camera_R", "camera_C")}
    camera.update(coordinate_system="CAD_metres", convention="pixel ~ K R (X-C)", image_size=a["camera_image_size"],
                  source=a["resources"]["fusion_archive"])
    save_json(output / "cameras.json", camera)
    save_json(output / "transform.json", {"schema": "zip2cad_local_normalization_v1", "origin_CAD_m": origin.tolist(),
              "scale_m": scale, "rotation": np.eye(3).tolist(), "forward": "(X_CAD-origin_CAD_m)/scale_m",
              "inverse": "X_local*scale_m+origin_CAD_m", "maximum_roundtrip_error_m": roundtrip})
    report = {"schema": "zip2cad_zipmap_adapter_report_v1", "status": "passed", "pointcloud_source": "ZipMap",
              "audit_directory": str(Path(audit_path).resolve()), "source_archive": a["resources"]["fusion_archive"],
              "point_count": len(p), "actual_contributing_source_frames": int(len(np.unique(refs))),
              "normal_valid_count": int(valid.sum()), "normal_valid_fraction": float(valid.mean()),
              "crop_z_bounds_m": bounds, "saved_building_PLY_exact_order_xyz_RGB_match": True,
              "all_point_attributes_synchronously_cropped": True, "new_similarity_applied": False,
              "support_count_is_probability": False, "source_frame_is_full_support_list": False,
              "local_roundtrip_error_m": roundtrip, "normal_policy": a["normals"],
              "independent_image_holdout": False, "Gaussian_resources_required": False}
    report["output_hashes"] = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}
    save_json(output / "adapter_report.json", report)
    return report


def check_adapter(path):
    path = Path(path).resolve()
    report = read_json(path / "adapter_report.json")
    a, root = checked_audit(report["audit_directory"])
    inside(root, path)
    if report["status"] != "passed" or report["pointcloud_source"] != "ZipMap":
        raise ValueError("Successful ZipMap adapter required")
    for name, digest in report["output_hashes"].items():
        if Path(name).name != name or sha256_file(path / name) != digest:
            raise ValueError("Adapted data changed: " + name)
    return a, root, report


def suppress_planar_duplicates(components, rows, tolerance):
    """Suppress near-coincident surfaces; parallel separated levels survive."""
    from shapely.geometry import Polygon
    rows_by_id = {r["component_id"]: r for r in rows}
    selected = []
    for part in sorted(components, key=lambda p: -rows_by_id[p["component_id"]]["area_m2"]):
        row = rows_by_id[part["component_id"]]
        n = np.asarray(row["plane_normal"])
        o = np.asarray(row["plane_origin"])
        top = np.asarray(part["vertices_world_CAD_m"])[::1]
        # polygon_prism stores top and bottom vertices in equal halves.
        top = top[:len(top) // 2]
        axis = np.eye(3)[np.argmin(abs(n))]
        u = np.cross(n, axis); u /= np.linalg.norm(u)
        basis = np.c_[u, np.cross(n, u)]
        footprint = Polygon((top - o) @ basis)
        duplicate = None
        for previous in selected:
            old = rows_by_id[previous["component_id"]]
            if abs(n @ old["plane_normal"]) < np.cos(np.deg2rad(8)):
                continue
            old_top = np.asarray(previous["vertices_world_CAD_m"])
            old_top = old_top[:len(old_top) // 2]
            if np.median(abs((old_top - o) @ n)) > tolerance:
                continue
            old_poly = Polygon((old_top - o) @ basis)
            if footprint.intersection(old_poly).area / max(min(footprint.area, old_poly.area), 1e-9) > .6:
                duplicate = previous["component_id"]
                break
        if duplicate:
            row.update(accepted=False, native_edit_approved=False, rejection_reason="near_coincident_duplicate", duplicate_of=duplicate)
        else:
            selected.append(part)
    return selected


def reject_unoriented_plane_slices(rows, sample_normals, normal_valid, minimum_fraction):
    """A RANSAC slab through a noisy volume is not a measured surface.

    Evaluate independent local-PCA orientation on the original supporting
    observations, not on synthetic points sampled from the fitted plane.
    """
    for row in rows:
        ids = row["training_observation_indices"]
        aligned = normal_valid[ids] & (abs(sample_normals[ids] @ row["plane_normal"]) >= np.cos(np.deg2rad(30)))
        fraction = float(np.mean(aligned)) if len(ids) else 0.
        row["reliable_aligned_normal_fraction"] = fraction
        row["minimum_reliable_aligned_normal_fraction"] = minimum_fraction
        if row["accepted"] and fraction < minimum_fraction:
            row.update(accepted=False, native_edit_approved=False, rejection_reason="RANSAC_slice_lacks_local_surface_orientation")


def build_baseline(adapter_path, output, maximum_fit_points=40000):
    """Create a bounded, reproducible geometry-only baseline, not a trained model."""
    import shutil
    from .structural_surfaces import SurfacePolicy, extract_structural_surfaces
    from .native_skp import write_native
    from .model_preview import preview_models, render_mesh

    a, root, adapted = check_adapter(adapter_path)
    if type(maximum_fit_points) is not int or maximum_fit_points < 500:
        raise ValueError("At least 500 fitting point budget required")
    adapter_path = Path(adapter_path).resolve()
    with np.load(adapter_path / "observed.npz", allow_pickle=False) as z:
        p, refs = z["points_CAD_m"], z["frame_refs"]
        normal, normal_valid = z["normal"], z["normal_valid"]
    from shapely.geometry import Polygon
    context = a["CAD_context"]
    target = Polygon(context["CAD_target_polygon_xy"])
    base = context["CAD_attachment_z_m"]
    settings = a["baseline"]
    roof = contains_xy(target, p[:, 0], p[:, 1]) & (p[:, 2] > base + settings["minimum_height_above_support_m"]) & (p[:, 2] < base + settings["maximum_height_above_support_m"])
    candidates = np.flatnonzero(roof)
    if len(candidates) < 100:
        raise ValueError("Insufficient observed elevated roof points")
    # A deterministic random subset of actual observations. It is not a
    # resampled CAD surface, interpolated cloud, or a synthetic training set.
    rng = np.random.default_rng(20261015)
    selected = np.sort(rng.choice(candidates, min(maximum_fit_points, len(candidates)), replace=False))
    fit, fit_refs = p[selected], refs[selected]
    all_refs = sorted(map(int, np.unique(fit_refs)))
    check = all_refs[::a["internal_check_stride"]]
    training = sorted(set(all_refs) - set(check))
    with np.load(a["source_cad"], allow_pickle=False) as z:
        cad_data = {k: z[k] for k in z.files}
    prefix = "skp:" + a["project_id"] + ":" + ".".join(map(str, a["scope_pid_path"])) + "."
    own = np.array([str(x).startswith(prefix) for x in cad_data["entity_ids"]])
    mesh = trimesh.Trimesh(cad_data["vertices_world_m"], cad_data["faces"][own[cad_data["face_owner_indices"]]], process=False)
    mesh.update_faces(mesh.nondegenerate_faces())
    policy = SurfacePolicy(**{k: v for k, v in settings.items() if k in SurfacePolicy.__dataclass_fields__})
    output = fresh_output(root, output, [adapter_path, a["source_cad"], a["resources"]["source_model"]["path"]])
    print("Finite surface fit", len(fit), "actual roof observations", flush=True)
    extracted = extract_structural_surfaces(fit, fit_refs, mesh, training, check, policy)
    reject_unoriented_plane_slices(extracted["candidate_rows"], normal[selected], normal_valid[selected], settings.get("minimum_reliable_normal_fraction", .25))
    accepted_ids = {row["component_id"] for row in extracted["candidate_rows"] if row["accepted"]}
    extracted["components"] = [part for part in extracted["components"] if part["component_id"] in accepted_ids]
    for row in extracted["candidate_rows"]:
        row["native_edit_approved"] = bool(row["accepted"])
        row["adapted_observation_indices"] = selected[row["training_observation_indices"]].tolist()
        row.update(validation_scope="internal_source_frame_partition_of_pre_fused_cloud", independent_image_validation=False,
                   pointcloud_source="ZipMap", physical_truth_verified=False)
    parts = suppress_planar_duplicates(extracted["components"], extracted["candidate_rows"], max(.25, 2 * policy.plane_tolerance_m))
    extracted["components"] = parts
    np.savez_compressed(output / "fit_source_indices.npz", adapted_indices=selected)
    save_json(output / "components.json", {"schema": "zip2cad_zipmap_baseline_components_v1", **extracted,
              "pointcloud_source": "ZipMap", "audit_directory": adapted["audit_directory"], "adapter_directory": str(adapter_path),
              "Gaussian_resources_required": False, "source_model": a["resources"]["source_model"],
              "policy": asdict(policy), "truth_labels": False})
    save_json(output / "pointcloud_completion_report.json", {"components": extracted["candidate_rows"]})
    if parts:
        plan = {"schema": "zip2cad_native_completion_plan_v2", "source_model": a["resources"]["source_model"]["path"],
                "source_model_sha256": a["resources"]["source_model"]["sha256"], "components": parts,
                "textures_allowed": False, "hide_existing_pid_paths": [], "view_scope_pid_path": a["scope_pid_path"],
                "evidence_report_sha256": sha256_file(output / "pointcloud_completion_report.json")}
        save_json(output / "native_plan.json", plan)
        receipt = write_native(output / "native_plan.json", output / "native")
        target_file = Path(receipt["updated_model"])
        preview_models(plan["source_model"], target_file, a["scope_pid_path"], receipt["generated_assembly_pid"], output / "preview")
        status = "geometry_baseline_saved_and_reloaded"
    else:
        target_file = Path(a["resources"]["source_model"]["path"])
        from .sketchup_sdk import SketchUpSDK
        with SketchUpSDK() as sdk:
            model = sdk.load(target_file)
            try:
                before = sdk.geometry(model, scope=a["scope_pid_path"], include_appearance=True)
            finally:
                sdk.release(model)
        (output / "preview").mkdir()
        render_mesh(before, output / "preview/after.png", "ZipMap baseline: no supported addition")
        status = "no_supported_addition_baseline_retained"
    shutil.copy2(target_file, output / "zipmap_coarse.skp")
    # Residual evaluation is a geometric agreement diagnostic, not an accuracy
    # score or a roof coverage certificate. Keep original sample indices.
    assigned = np.full(len(fit), -1, np.int32)
    best = np.full(len(fit), np.inf)
    for i, part in enumerate(parts):
        part_mesh = trimesh.Trimesh(part["vertices_world_CAD_m"], part["triangles"], process=False)
        distances = trimesh.proximity.closest_point(part_mesh, fit)[1]
        better = distances < best
        best[better], assigned[better] = distances[better], i
    supported = best <= policy.plane_tolerance_m * 1.5
    assigned[~supported] = -1
    np.savez_compressed(output / "baseline_assignments.npz", adapted_indices=selected, proposed_instance=assigned,
                        distance_to_proposal_m=best)
    metrics = {"schema": "zip2cad_zipmap_baseline_metrics_v1", "status": status, "pointcloud_source": "ZipMap",
               "adapter_directory": str(adapter_path), "candidate_count": len(extracted["candidate_rows"]),
               "native_component_count": len(parts), "elevated_roof_observation_count": len(candidates), "fit_observation_count": len(fit),
               "seed": 20261015, "training_source_frames": training, "internal_check_source_frames": check,
               "independent_image_validation": False, "neural_model_used": False, "Gaussian_resources_required": False,
               "normal_valid_fraction": adapted["normal_valid_fraction"], "local_orientation_evidence_gate": True,
               "minimum_reliable_normal_fraction": settings.get("minimum_reliable_normal_fraction", .25),
               "observed_sample_near_proposal_fraction": float(supported.mean()), "whole_roof_truth_coverage": None,
               "output_model": str(output / "zipmap_coarse.skp"), "output_model_sha256": sha256_file(output / "zipmap_coarse.skp"),
               "source_file_unchanged": sha256_file(a["resources"]["source_model"]["path"]) == a["resources"]["source_model"]["sha256"],
               "unresolved_scope": ["Nonplanar and small roof details", "Occluded structure", "Possible convex-boundary infill", "Raw cloud thickness and systematic geometric bias"],
               "baseline_is_ground_truth": False, "all_missing_building_parts_completed": False}
    save_json(output / "baseline_metrics.json", metrics)
    save_json(output / "baseline_manifest.json", {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()})
    return metrics


def prepare_review(baseline_path, output):
    """Prepare evidence for human review without manufacturing training truth."""
    import csv
    import html
    import cv2
    from PIL import Image, ImageDraw
    baseline_path = Path(baseline_path).resolve()
    metrics = read_json(baseline_path / "baseline_metrics.json")
    a, root, adapted = check_adapter(metrics["adapter_directory"])
    inside(root, baseline_path)
    for name, digest in read_json(baseline_path / "baseline_manifest.json").items():
        if Path(name).name != name or sha256_file(baseline_path / name) != digest:
            raise ValueError("Baseline changed: " + name)
    if metrics["pointcloud_source"] != "ZipMap":
        raise ValueError("ZipMap baseline required")
    components = read_json(baseline_path / "components.json")
    output = fresh_output(root, output, [baseline_path, a["resources"]["source_video"]["path"]])
    images_path = output / "evidence"
    images_path.mkdir()
    n = adapted["point_count"]
    np.savez_compressed(output / "labels.npz", semantic=np.full(n, -1, np.int16),
                        instance_id=np.full(n, -1, np.int32), label_valid=np.zeros(n, bool))
    camera = read_json(Path(metrics["adapter_directory"]) / "cameras.json")
    positions = {int(ref): i for i, ref in enumerate(camera["camera_refs"])}
    K, R, C = (np.asarray(camera[k]) for k in ("camera_K", "camera_R", "camera_C"))
    frame_manifest = read_json(a["resources"]["frame_manifest"]["path"])
    frames = {int(row["accepted_index"] - 1): row for row in frame_manifest["frames"] if row["status"] == "accepted"}
    video = cv2.VideoCapture(a["resources"]["source_video"]["path"])
    if not video.isOpened():
        raise ValueError("Cannot decode source video")
    native_size = [int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    cache, evidence, targets, worksheet = {}, [], [], []
    cards = []
    by_id = {r["component_id"]: r for r in components["candidate_rows"]}
    try:
        for part in components["components"]:
            name = part["component_id"]
            row = by_id[name]
            refs = sorted(set(row["training_frame_refs"]))
            chosen = [refs[i] for i in sorted(set([0, len(refs) // 2, len(refs) - 1]))]
            component_images = []
            vertices = np.asarray(part["vertices_world_CAD_m"])
            triangles = np.asarray(part["triangles"], int)
            for ref in chosen:
                if ref not in cache:
                    video.set(cv2.CAP_PROP_POS_FRAMES, frames[ref]["source_frame_index"])
                    ok, image = video.read()
                    if not ok:
                        raise ValueError("Failed to decode source frame " + str(ref))
                    cache[ref] = image
                image = cache[ref].copy()
                i = positions[ref]
                cam = (vertices - C[i]) @ R[i].T
                projection = cam @ K[i].T
                uv = projection[:, :2] / np.maximum(projection[:, 2:], 1e-9)
                uv *= np.asarray(native_size) / camera["image_size"]
                for tri in triangles:
                    if np.all(cam[tri, 2] > 0):
                        screen = np.clip(uv[tri], -100000, 100000).astype(np.int32)
                        cv2.polylines(image, [screen], True, (0, 80, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"{name} | source ref {ref} | PROPOSAL NOT GROUND TRUTH", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)
                image = cv2.resize(image, (960, int(960 * image.shape[0] / image.shape[1])))
                file = images_path / f"{name}_ref_{ref}.jpg"
                if not cv2.imwrite(str(file), image):
                    raise ValueError("Cannot save review image")
                record = {"component_id": name, "frame_ref": ref, "source_frame_index": frames[ref]["source_frame_index"],
                          "timestamp_s": frames[ref]["timestamp_s"], "overlay": file.relative_to(output).as_posix(),
                          "projection_is_visual_review_only": True, "occlusion_first_hit_verified": False}
                evidence.append(record)
                component_images.append(record["overlay"])
            targets.append({"component_id": name, "proposed_geometry_type": row["surface_type"],
                            "vertices_world_CAD_m": part["vertices_world_CAD_m"], "triangles": part["triangles"],
                            "target_valid": False, "truth_grade": "unreviewed", "source": "ZipMap_B0_proposal_not_truth",
                            "plane_origin": row["plane_origin"], "plane_normal": row["plane_normal"],
                            "review_evidence_frame_refs": chosen, "verified_connections": [], "connection_labels_valid": False})
            worksheet.append(dict(component_id=name, proposed_type=row["surface_type"], area_m2=round(row["area_m2"], 3),
                                  evidence_refs=" ".join(map(str, chosen)), review_action="unreviewed",
                                  corrected_type="", boundary_confirmed="", connections_confirmed="", truth_grade="", notes=""))
            images = "".join(f'<a href="{f}"><img src="{f}" alt="Source video evidence for {name}"></a>' for f in component_images)
            cards.append(f'<section><h2>{html.escape(name)} · {html.escape(row["surface_type"])} · {row["area_m2"]:.2f} m²</h2><p>待核验的几何提议。红色投影仅用于定位，尚未验证遮挡和真实边界。</p><div class="images">{images}</div></section>')
    finally:
        video.release()
    if not worksheet:
        worksheet.append(dict(component_id="whole_roof_review", proposed_type="unknown", area_m2="",
                              evidence_refs="", review_action="unreviewed", corrected_type="", boundary_confirmed="",
                              connections_confirmed="", truth_grade="", notes="需要已核验的屋顶参考SKP或构件标注，不能使用被拒绝面片作标签"))
    # Even when all proposals are rejected, preserve a usable raw-video review
    # pack. These images are context, never confirmation that a proposal exists.
    context_refs = [int(camera["camera_refs"][i]) for i in np.linspace(0, len(camera["camera_refs"]) - 1, 8, dtype=int)]
    video = cv2.VideoCapture(a["resources"]["source_video"]["path"])
    context_images = []
    try:
        for ref in context_refs:
            video.set(cv2.CAP_PROP_POS_FRAMES, frames[ref]["source_frame_index"])
            ok, image = video.read()
            if not ok:
                raise ValueError("Cannot decode roof context frame")
            image = cv2.resize(image, (960, int(960 * image.shape[0] / image.shape[1])))
            cv2.putText(image, f"SOURCE VIDEO ref={ref} | NO GENERATED GEOMETRY", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 255), 2)
            file = images_path / f"roof_context_ref_{ref}.jpg"
            if not cv2.imwrite(str(file), image):
                raise ValueError("Cannot write context image")
            rel = file.relative_to(output).as_posix()
            context_images.append(rel)
            evidence.append(dict(component_id="roof_context", frame_ref=ref, source_frame_index=frames[ref]["source_frame_index"],
                                 timestamp_s=frames[ref]["timestamp_s"], overlay=rel, context_only=True))
    finally:
        video.release()
    cards.insert(0, '<section><h2>原视频代表视角</h2><p>用于确认参考模型和需要补拍的位置，未绘入生成几何。</p><div class="images">' + ''.join(f'<a href="{f}"><img src="{f}" alt="原视频代表视角"></a>' for f in context_images) + '</div></section>')
    with (output / "review_worksheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        columns = ["component_id", "proposed_type", "area_m2", "evidence_refs", "review_action", "corrected_type", "boundary_confirmed", "connections_confirmed", "truth_grade", "notes"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader(); writer.writerows(worksheet)
    save_json(output / "target_components.json", {"schema": "zip2cad_unverified_targets_v1", "components": targets,
              "pointcloud_source": "ZipMap", "valid_target_count": 0, "training_ready": False})
    save_json(output / "truth_evidence.json", {"frame_evidence": evidence, "verified_by_human": False,
              "image_size_scaling": {"camera": camera["image_size"], "native": native_size},
              "projection_first_hit_verified": False, "all_labels_ignored_until_review": True})
    with (output / "capture_gaps.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["region", "reason", "requested_evidence", "status"])
        for region, reason, request in [
            ("unassigned_roof_samples", "B0仅覆盖部分平面，未建完整构件清单", "检查原视频是否有缺失构件清晰多视角；不足时补拍"),
            ("roof_connections", "连接真值尚未核验", "确认端点真实相连或真实间隙，并标出对应图像"),
            ("reference_SKP", "图3尚无已核验的对应原生模型", "提供项目目录内参考SKP的准确路径并说明屋顶是否与现场一致"),
        ]:
            writer.writerow([region, reason, request, "pending_review"])
    title = "ZipMap 屋顶训练标签核验"
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{title}</title>
<style>body{{font:16px/1.7 'Microsoft YaHei',sans-serif;max-width:1400px;margin:32px auto;padding:0 24px;color:#202020}}h1,h2{{line-height:1.35}}section{{margin:38px 0}}.images{{display:flex;gap:12px;flex-wrap:wrap}}.images a{{width:31%}}img{{width:100%;height:auto}}code{{font-size:14px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:9px}}</style>
<h1>{title}</h1><p>仅使用 ZipMap 点云与原视频。此页用于准备人工核验，未训练神经网络。</p>
<p>本次B0写入 {len(targets)} 个有限平面实体。它们只用于构件定位，不能直接作为完整正确屋顶。</p>
<p><a href="review_worksheet.csv">打开待核验CSV</a> · <a href="capture_gaps.csv">证据缺口清单</a> · <a href="target_components.json">未核验构件JSON</a></p>
<h2>需要核对的内容</h2><ol><li>查看各构件的红色定位投影，判断该区域是否存在对应构件。</li><li>在CSV的review_action填写 confirm_visible_geometry、incorrect_geometry 或 insufficient_evidence；在notes写明边界、形状和连接问题。</li><li>仅看到位置不能确认尺寸和厚度；不能确认时保持未核验。确认结果先用于修订标注，程序不会直接把CSV确认变成全部点的正确标签。</li><li>若有图3对应原生SKP，把文件放在项目数据目录内并在配置中声明路径；先核验再作真值。</li></ol>
<p>当前标签文件中 {n:,} 个点均为未知，label_valid全部为false。必须完成有依据的标注后才可训练真实屋顶监督任务。法向可靠点比例约 {adapted['normal_valid_fraction']:.1%}，其他点保留无效掩膜。</p>
{''.join(cards)}<h2>未覆盖区域</h2><p>B0未检出不代表现场不存在。继续检查条带、竖墙、紧凑体及连接，原始可信外壳保持保护。</p></html>'''
    (output / "START_HERE.html").write_text(body, encoding="utf-8")
    if evidence:
        thumbnails = []
        for item in evidence:
            im = Image.open(output / item["overlay"]).convert("RGB")
            im.thumbnail((480, 270))
            thumbnails.append(im)
        sheet = Image.new("RGB", (1440, 300 * ((len(thumbnails) + 2) // 3)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, im in enumerate(thumbnails):
            x, y = (i % 3) * 480, (i // 3) * 300
            sheet.paste(im, (x, y))
            draw.text((x + 8, y + 275), f"{evidence[i]['component_id']} ref={evidence[i]['frame_ref']}", fill="black")
        sheet.save(output / "review_contact_sheet.jpg", quality=90)
    result = {"schema": "zip2cad_review_preparation_v1", "status": "awaiting_reference_and_label_review", "step4_complete": False,
              "baseline_directory": str(baseline_path), "adapter_directory": metrics["adapter_directory"],
              "point_count": n, "valid_label_count": 0, "proposed_component_count": len(targets),
              "evidence_image_count": len(evidence), "training_ready": False, "human_action_required": True,
              "source_video_unchanged": sha256_file(a["resources"]["source_video"]["path"]) == a["resources"]["source_video"]["sha256"]}
    save_json(output / "review_status.json", result)
    return result
