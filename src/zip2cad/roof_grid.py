"""Roof top-surface rasterisation in unchanged CAD metre coordinates.

This 2.5D supervision branch excludes covered lower surfaces by depth buffering.
It cannot represent multiple visible levels at the same XY, walls or interiors.
"""
import numpy as np


def raster_top(vertices, faces, triangle_class, xy_origin, cell_m, shape):
    if cell_m <= 0 or min(shape) <= 0:
        raise ValueError("Positive raster resolution and shape required")
    vertices = np.asarray(vertices, float)
    faces = np.asarray(faces, int)
    if not np.isfinite(vertices).all():
        raise ValueError("Nonfinite CAD vertices")
    height = np.full(shape, -np.inf, np.float64)
    label = np.full(shape, -1, np.int16)
    owner = np.full(shape, -1, np.int32)
    origin = np.asarray(xy_origin, float)
    for i, tri in enumerate(vertices[faces]):
        uv = (tri[:, :2] - origin) / cell_m - .5
        lo = np.maximum(np.ceil(uv.min(0)).astype(int), 0)
        hi = np.minimum(np.floor(uv.max(0)).astype(int), [shape[1]-1, shape[0]-1])
        if np.any(hi < lo):
            continue
        a, b, c = uv
        den = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(den) < 1e-10:
            continue
        y, x = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        l1 = ((b[1]-c[1])*(x-c[0]) + (c[0]-b[0])*(y-c[1])) / den
        l2 = ((c[1]-a[1])*(x-c[0]) + (a[0]-c[0])*(y-c[1])) / den
        l3 = 1-l1-l2
        z = l1*tri[0, 2]+l2*tri[1, 2]+l3*tri[2, 2]
        ys, xs = slice(lo[1], hi[1]+1), slice(lo[0], hi[0]+1)
        h = height[ys, xs]
        keep = (l1 >= -1e-8) & (l2 >= -1e-8) & (l3 >= -1e-8) & (z > h + 1e-8)
        h[keep] = z[keep]
        label[ys, xs][keep] = triangle_class[i]
        owner[ys, xs][keep] = i
    valid = np.isfinite(height)
    height[~valid] = 0
    return height.astype(np.float32), label, valid, owner


def observed_grid(points, colours, refs, xy_origin, cell_m, shape, base_z, scale_m=6.):
    p = np.asarray(points, float)
    ij = np.floor((p[:, :2] - xy_origin) / cell_m).astype(int)
    valid = (ij[:, 0] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[0])
    p, ij, rgb, refs = p[valid], ij[valid], colours[valid], refs[valid]
    flat = ij[:, 1]*shape[1] + ij[:, 0]
    n = int(np.prod(shape))
    count = np.bincount(flat, minlength=n)
    z = (p[:, 2] - base_z) / scale_m
    mean = np.bincount(flat, weights=z, minlength=n)/np.maximum(count, 1)
    var = np.bincount(flat, weights=z*z, minlength=n)/np.maximum(count, 1)-mean*mean
    low, high = np.full(n, np.inf), np.full(n, -np.inf)
    np.minimum.at(low, flat, z); np.maximum.at(high, flat, z)
    low[count == 0] = 0; high[count == 0] = 0
    pairs = np.unique(np.c_[flat, refs.astype(np.int64)], axis=0)
    frame_count = np.bincount(pairs[:, 0], minlength=n)
    channels = [low, mean, high, np.sqrt(np.maximum(var, 0)),
                np.clip(np.log1p(count)/np.log(1001), 0, 1),
                np.clip(np.log1p(frame_count)/np.log(101), 0, 1)]
    channels += [np.bincount(flat, weights=rgb[:, k]/255, minlength=n)/np.maximum(count, 1) for k in range(3)]
    channels.append((count >= 3).astype(float))
    return np.stack(channels).reshape(10, *shape).astype(np.float32), count.reshape(shape), frame_count.reshape(shape)
