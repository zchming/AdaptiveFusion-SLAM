"""Sufficient rejection witnesses, never a full BA observability certificate.

Single-landmark separators are found in the camera/landmark incidence graph.
Two-point and collinear support is checked across ordered temporal cuts only.
The latter does not cover arbitrary non-temporal multi-landmark separators.
"""
import time
import numpy as np

RELATIVE_COLLINEAR_TOLERANCE = 1e-8
ABSOLUTE_SPREAD_TOLERANCE_M = 1e-10


def single_landmark_separators(kids, pids, observations):
    """Iterative Tarjan articulation audit; no recursion or external graph package."""
    cameras = {kid: index for index, kid in enumerate(kids)}
    points = {pid: len(kids)+index for index, pid in enumerate(pids)}
    graph = [[] for _ in range(len(kids)+len(pids))]
    for observation in observations:
        a, b = cameras[observation.keyframe_id], points[observation.point_id]
        graph[a].append(b); graph[b].append(a)
    discovery = [-1]*len(graph); low = [0]*len(graph)
    parent = [-1]*len(graph); camera_count = [0]*len(graph)
    witnesses = {}; clock = 0
    for root in range(len(kids)):
        if discovery[root] != -1:
            continue
        discovery[root] = low[root] = clock; clock += 1
        camera_count[root] = 1
        stack = [(root, iter(graph[root]))]
        while stack:
            node, children = stack[-1]
            other = next(children, None)
            if other is None:
                stack.pop(); previous = parent[node]
                if previous != -1:
                    low[previous] = min(low[previous], low[node])
                    camera_count[previous] += camera_count[node]
                    # DFS roots are cameras. A point node is never a root;
                    # every separating child subtree contains a camera.
                    if previous >= len(kids) and low[node] >= discovery[previous]:
                        pid = pids[previous-len(kids)]
                        witnesses.setdefault(pid, []).append({
                            'child_keyframe': kids[node],
                            'separated_camera_count': camera_count[node]})
            elif discovery[other] == -1:
                parent[other] = node
                discovery[other] = low[other] = clock; clock += 1
                camera_count[other] = int(other < len(kids))
                stack.append((other, iter(graph[other])))
            elif other != parent[node]:
                low[node] = min(low[node], discovery[other])
    return [{'point_id': pid, 'separating_subtrees': witnesses[pid]}
            for pid in sorted(witnesses)]


def bridge_audit(problem):
    """Read-only active-problem audit; metric coordinates are in metres.

    One/two-landmark separators provide structural rotational-gauge witnesses.
    Collinearity of >=3 estimated landmarks is a conservative input-state
    degeneracy test, not proof that all possible measurement-consistent states
    are degenerate. Passing this finite set of tests is not full rank evidence.
    """
    start = time.perf_counter()
    if problem.begin != 0 or problem.boundary != 'window':
        raise ValueError('bridge audit requires a full batch window')
    kids, pids = problem.kids, problem.pids
    positions = {kid: index for index, kid in enumerate(kids)}
    ranges = {pid: [len(kids), -1] for pid in pids}
    for observation in problem.obs:
        index = positions[observation.keyframe_id]
        bounds = ranges[observation.point_id]
        bounds[0] = min(bounds[0], index); bounds[1] = max(bounds[1], index)
    starts, ends = {}, {}
    for pid, (first, last) in ranges.items():
        if first < last:
            starts.setdefault(first+1, []).append(pid)
            ends.setdefault(last+1, []).append(pid)
    active, cuts = set(), []
    for index in range(1, len(kids)):
        active.update(starts.get(index, [])); active.difference_update(ends.get(index, []))
        point_ids = sorted(active); count = len(point_ids)
        singular = []; kind = None
        if count < 3:
            kind = 'disconnected_cut' if count == 0 else 'at_most_two_shared_landmarks'
        else:
            xyz = np.array([problem.s.points[pid].xyz for pid in point_ids])
            singular = np.linalg.svd(xyz-xyz.mean(axis=0), compute_uv=False).tolist()
            if (singular[0] <= ABSOLUTE_SPREAD_TOLERANCE_M or
                    singular[1] <= RELATIVE_COLLINEAR_TOLERANCE*singular[0]):
                kind = 'numerically_collinear_at_input'
        cut = {'left_last_keyframe': kids[index-1], 'right_first_keyframe': kids[index],
               'crossing_landmarks': count, 'singular_values_m': singular, 'rejection_kind': kind}
        if kind:
            cut['point_ids'] = point_ids
        cuts.append(cut)
    separators = single_landmark_separators(kids, pids, problem.obs)
    failed = [cut for cut in cuts if cut['rejection_kind']]
    return {'decision': 'reject' if separators or failed else 'not_rejected_not_certified',
            'single_landmark_separators': separators,
            'temporal_cuts': cuts, 'failed_temporal_cuts': len(failed),
            'relative_collinear_tolerance': RELATIVE_COLLINEAR_TOLERANCE,
            'absolute_spread_tolerance_m': ABSOLUTE_SPREAD_TOLERANCE_M,
            'limitations': 'No full numerical rank certificate; arbitrary non-temporal two-point/collinear separators may be missed. Estimated-coordinate collinearity is input-state dependent.',
            'audit_ms': (time.perf_counter()-start)*1000}
