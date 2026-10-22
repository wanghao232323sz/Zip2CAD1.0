"""Seeded surface evaluation. No alignment or target-based candidate selection."""
import numpy as np
import trimesh


def surface_sample(mesh, count=20000, seed=20261020):
    areas = mesh.area_faces
    if not len(areas) or areas.sum() <= 0:
        raise ValueError('Nonempty surface required')
    rng = np.random.default_rng(seed)
    face = rng.choice(len(areas), count, p=areas / areas.sum())
    uv = rng.random((count, 2))
    uv[uv.sum(1) > 1] = 1 - uv[uv.sum(1) > 1]
    tri = mesh.triangles[face]
    return tri[:, 0] + uv[:, :1]*(tri[:, 1]-tri[:, 0]) + uv[:, 1:]*(tri[:, 2]-tri[:, 0])


def surface_distances(mesh, points, batch=1024):
    proximity = trimesh.proximity.ProximityQuery(mesh)
    return np.concatenate([proximity.on_surface(points[i:i+batch])[1]
                           for i in range(0, len(points), batch)])


def distance_summary(values):
    values = np.asarray(values)
    return dict(mean_m=float(values.mean()), median_m=float(np.median(values)),
                p90_m=float(np.quantile(values, .9)), rmse_m=float(np.sqrt(np.mean(values**2))))


def compare_surfaces(candidate, reference, reference_samples=None, count=20000, seed=20261020):
    q = surface_sample(reference, count, seed) if reference_samples is None else reference_samples
    p = surface_sample(candidate, count, seed)
    forward, reverse = surface_distances(reference, p), surface_distances(candidate, q)
    thresholds = []
    for tolerance in [.1, .25, .5, 1.]:
        precision = float(np.mean(forward <= tolerance))
        recall = float(np.mean(reverse <= tolerance))
        thresholds.append(dict(tolerance_m=tolerance, precision=precision, recall=recall,
                               F1=2*precision*recall/max(precision+recall, 1e-15)))
    return dict(candidate_to_reference=distance_summary(forward),
                reference_to_candidate=distance_summary(reverse),
                symmetric_mean_distance_m=float((forward.mean()+reverse.mean())/2),
                thresholds=thresholds, sample_count_per_surface=len(p)), forward, reverse
