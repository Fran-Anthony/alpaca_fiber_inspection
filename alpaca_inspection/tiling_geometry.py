"""Geometría compartida por la inferencia y las herramientas de dataset."""
import copy
from collections import defaultdict


def positions(length, size, overlap):
    if length < size:
        raise ValueError(f'Dimensión {length} menor que tile {size}; no se rellena ni escala.')
    result = list(range(0, length - size + 1, size - overlap))
    if result[-1] != length - size:
        result.append(length - size)
    return result


def bbox_iou_and_iomin(b1, b2):
    """Calcula (IoU, IoMin) entre dos cajas en formato COCO [x, y, w, h]."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0, 0.0
    a1 = max(w1 * h1, 1e-9)
    a2 = max(w2 * h2, 1e-9)
    union = a1 + a2 - inter
    return inter / union, inter / min(a1, a2)


def clip_bbox(bbox, max_w, max_h):
    """Recorta [x, y, w, h] a los límites [0, max_w] x [0, max_h]."""
    x, y, w, h = bbox
    x1 = max(0.0, min(float(max_w), x))
    y1 = max(0.0, min(float(max_h), y))
    x2 = max(0.0, min(float(max_w), x + w))
    y2 = max(0.0, min(float(max_h), y + h))
    nw = round(max(0.0, x2 - x1), 2)
    nh = round(max(0.0, y2 - y1), 2)
    return [round(x1, 2), round(y1, 2), nw, nh]


def merge_cluster(cluster, mode, img_w, img_h):
    """Combina un grupo de anotaciones solapadas provenientes de distintos tiles."""
    if len(cluster) == 1 or mode == 'none':
        single = copy.deepcopy(cluster[0])
        single['extra'] = {
            'merged_count': 1,
            'source_tiles': [single['_tile_coord']],
        }
        return single

    best = max(cluster, key=lambda a: a['bbox'][2] * a['bbox'][3])
    merged = copy.deepcopy(best)
    scored = all('score' in ann for ann in cluster)

    if mode == 'largest':
        merged['bbox'] = clip_bbox(best['bbox'], img_w, img_h)
    elif mode == 'union':
        x1 = min(a['bbox'][0] for a in cluster)
        y1 = min(a['bbox'][1] for a in cluster)
        x2 = max(a['bbox'][0] + a['bbox'][2] for a in cluster)
        y2 = max(a['bbox'][1] + a['bbox'][3] for a in cluster)
        merged['bbox'] = clip_bbox([x1, y1, x2 - x1, y2 - y1], img_w, img_h)
    elif mode == 'wbf':
        if scored:
            weights = [max(a['score'], 1e-9) for a in cluster]
        else:
            weights = [max(a['bbox'][2] * a['bbox'][3], 1.0) for a in cluster]
        w_sum = sum(weights)
        x1 = sum(a['bbox'][0] * w for a, w in zip(cluster, weights)) / w_sum
        y1 = sum(a['bbox'][1] * w for a, w in zip(cluster, weights)) / w_sum
        x2 = sum((a['bbox'][0] + a['bbox'][2]) * w for a, w in zip(cluster, weights)) / w_sum
        y2 = sum((a['bbox'][1] + a['bbox'][3]) * w for a, w in zip(cluster, weights)) / w_sum
        merged['bbox'] = clip_bbox([x1, y1, x2 - x1, y2 - y1], img_w, img_h)
    else:
        raise ValueError(f'Modo de deduplicación no soportado: {mode}')

    if scored:
        strongest = max(cluster, key=lambda a: a['score'])
        merged['score'] = strongest['score']
        merged['category_id'] = strongest['category_id']

    merged['area'] = round(merged['bbox'][2] * merged['bbox'][3], 3)
    merged['extra'] = {
        'merged_count': len(cluster),
        'source_tiles': [a['_tile_coord'] for a in cluster],
    }
    return merged


def deduplicate_cross_tile_annotations(anns, mode, iou_thresh, iomin_thresh, img_w, img_h, class_agnostic=False):
    """Agrupa y fusiona anotaciones de la misma categoría que provienen de tiles distintos y se solapan."""
    if mode == 'none':
        out = []
        for a in anns:
            ac = copy.deepcopy(a)
            ac['extra'] = {'merged_count': 1, 'source_tiles': [ac['_tile_coord']]}
            out.append(ac)
        return out, 0

    if not anns:
        return [], 0

    n = len(anns)
    adj = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            if anns[i]['_tile_id'] == anns[j]['_tile_id']:
                continue
            if not class_agnostic and anns[i]['category_id'] != anns[j]['category_id']:
                continue
            iou, iomin = bbox_iou_and_iomin(anns[i]['bbox'], anns[j]['bbox'])
            if iou >= iou_thresh or iomin >= iomin_thresh:
                adj[i].add(j)
                adj[j].add(i)

    visited = [False] * n
    result = []
    merged_removed = 0

    for i in range(n):
        if visited[i]:
            continue
        comp = []
        queue = [i]
        visited[i] = True
        while queue:
            curr = queue.pop(0)
            comp.append(anns[curr])
            for nxt in sorted(adj[curr]):
                if not visited[nxt]:
                    visited[nxt] = True
                    queue.append(nxt)

        merged_ann = merge_cluster(comp, mode, img_w, img_h)
        result.append(merged_ann)
        if len(comp) > 1:
            merged_removed += len(comp) - 1

    return result, merged_removed
