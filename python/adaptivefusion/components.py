"""Shared-landmark graph topology; no new matches or inter-component constraints."""


def keyframe_components(snapshot, keyframe_ids, point_ids):
    """Include isolated cameras, using only the supplied eligible observations."""
    parents = {k: k for k in keyframe_ids}
    points = set(point_ids)

    def root(k):
        while parents[k] != k:
            parents[k] = parents[parents[k]]
            k = parents[k]
        return k

    first = {}
    for observation in snapshot.observations:
        k, p = observation.keyframe_id, observation.point_id
        if k not in parents or p not in points:
            continue
        if p in first:
            parents[root(k)] = root(first[p])
        else:
            first[p] = k
    groups = {}
    for k in sorted(parents):
        groups.setdefault(root(k), []).append(k)
    return sorted(groups.values(), key=lambda group: group[0])
