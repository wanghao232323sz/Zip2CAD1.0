"""Same-image depth correspondences, independent of CAD and roof assumptions."""
import numpy as np
from .similarity import estimate_similarity, transform_points


def unproject(depth, intrinsic, extrinsic, pixels):
    d, k, e, xy = map(np.asarray, (depth, intrinsic, extrinsic, pixels))
    rays = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(k).T
    camera = rays * (d[xy[:, 1], xy[:, 0]] / rays[:, 2])[:, None]
    return np.linalg.solve(e[:, :3], (camera-e[:, 3]).T).T


def robust_similarity(source, target, iterations=8, retain=.7):
    """Fixed trimmed fitting; held-out correspondences must not enter this call."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 30 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('at least 30 finite corresponding 3D points required')
    if not 0.5 <= retain <= 1 or iterations < 1:
        raise ValueError('invalid fixed trimming policy')
    if np.linalg.svd(source-source.mean(0), compute_uv=False)[1] < 1e-8:
        raise ValueError('collinear points do not constrain a spatial rotation')
    ids = np.arange(len(source))
    for _ in range(iterations):
        sim = estimate_similarity(source[ids], target[ids])
        if sim[0] <= 0:
            raise ValueError('nonpositive scale')
        errors = np.linalg.norm(transform_points(source, *sim)-target, axis=1)
        ids = np.argsort(errors, kind='stable')[:max(30, int(retain*len(source)))]
    return estimate_similarity(source[ids], target[ids])


def compose(parent, child):
    s, r, t = parent; cs, cr, ct = child
    return s*cs, r@cr, s*r@ct+t


def orientation_error(previous_extrinsic, current_extrinsic, current_to_previous):
    a, b = np.asarray(previous_extrinsic)[:, :3], np.asarray(current_extrinsic)[:, :3]
    difference = a @ current_to_previous @ b.T
    return float(np.degrees(np.arccos(np.clip((np.trace(difference)-1)/2, -1, 1))))
