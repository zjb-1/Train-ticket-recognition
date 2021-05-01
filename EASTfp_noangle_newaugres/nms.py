import numpy as np
from shapely.geometry import Polygon
import time


def intersection_me(g, p):
    min_xy = np.maximum(g[:2], p[:2])
    max_xy = np.minimum(g[4:6], p[4:6])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    inter_area = inter[0]*inter[1]

    g_area = (g[4] - g[0]) * (g[5] - g[1])
    p_area = (p[4] - p[0]) * (p[5] - p[1])
    union_area = g_area + p_area - inter_area

    if union_area == 0:
        return 0
    else:
        return inter_area/union_area


##   用 Polygon 模块时间 太慢
def intersection(g, p):
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))

    if not g.is_valid or not p.is_valid:
        return 0

    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union


def weighted_merge(g, p):
    g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])
    g[8] += p[8]
    return g


def standard_nms(S, thres):
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection_me(S[i], S[t]) for t in order[1:]])
        inds = np.where(ovr <= thres)[0]
        order = order[inds + 1]
    return S[keep]


def nms_locality(polys, thres=0.2):
    S = []        ##合并后的几何体集合
    p = None      #合并后的几何体
    for g in polys:
        if p is not None and intersection_me(g, p) > thres:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)
    if len(S) == 0:
        return np.array([])

    return standard_nms(np.array(S), thres)


