"""Versioned snapshot schema shared with C++; raw depths are immutable uint16 samples."""
from dataclasses import dataclass, field
from pathlib import Path
import copy
import hashlib
import numpy as np
from .geometry import check_pose


@dataclass
class Keyframe:
    id: int
    source_frame_id: int
    timestamp: float
    width: int
    height: int
    t_wc: np.ndarray


@dataclass
class Point:
    id: int
    host: int
    xyz: np.ndarray


@dataclass(frozen=True)
class Observation:
    point_id: int
    keyframe_id: int
    feature_index: int
    u: float
    v: float
    raw_depth: int


@dataclass
class Config:
    window_size: int = 5
    minimum_point_observations: int = 2
    maximum_iterations: int = 20
    huber_delta_pixels: float = 2.
    depth_scale: float = 5000.
    depth_residual_weight: float = 100.


@dataclass
class Snapshot:
    camera: np.ndarray
    keyframes: list
    points: dict
    observations: list
    config: Config = field(default_factory=Config)
    source_revision: str = 'synthetic'
    state_version: int = 0

    def clone(self):
        return copy.deepcopy(self)

    def validate(self):
        c = self.config
        if (np.shape(self.camera) != (4,) or not np.isfinite(self.camera).all()
                or np.any(np.asarray(self.camera)[:2] <= 0)):
            raise ValueError('invalid camera')
        if (not all(isinstance(v, int) for v in [c.window_size, c.minimum_point_observations,
                                                c.maximum_iterations, self.state_version])
                or c.window_size < 2 or c.minimum_point_observations < 2
                or c.maximum_iterations <= 0 or self.state_version < 0
                or not np.isfinite([c.huber_delta_pixels, c.depth_scale, c.depth_residual_weight]).all()
                or min(c.huber_delta_pixels, c.depth_scale, c.depth_residual_weight) <= 0):
            raise ValueError('invalid config/version')
        if not self.source_revision or any(x.isspace() for x in self.source_revision):
            raise ValueError('invalid source revision')
        previous_time, previous_frame = -np.inf, -1
        for i, k in enumerate(self.keyframes):
            check_pose(k.t_wc)
            if (k.id != i or not np.isfinite(k.timestamp) or k.timestamp <= previous_time
                    or k.source_frame_id <= previous_frame or min(k.width, k.height) <= 0
                    or max(k.width, k.height) > 4096):
                raise ValueError('invalid keyframe IDs, ordering or dimensions')
            previous_time, previous_frame = k.timestamp, k.source_frame_id
        for pid, p in self.points.items():
            if (pid != p.id or pid < 0 or not 0 <= p.host < len(self.keyframes)
                    or np.shape(p.xyz) != (3,) or not np.isfinite(p.xyz).all()):
                raise ValueError('invalid point')
        seen, features, hosts, samples = set(), set(), set(), {}
        for o in self.observations:
            if o.point_id not in self.points or not 0 <= o.keyframe_id < len(self.keyframes):
                raise ValueError('dangling observation')
            k = self.keyframes[o.keyframe_id]
            if (o.feature_index < 0 or o.feature_index > 100000 or not np.isfinite([o.u, o.v]).all()
                    or not 0 <= o.u < k.width or not 0 <= o.v < k.height
                    or float(np.float32(o.u)) != o.u or float(np.float32(o.v)) != o.v
                    or not isinstance(o.raw_depth, int) or not 0 <= o.raw_depth <= 65535):
                raise ValueError('invalid observation sample')
            pair, feature = (o.point_id, o.keyframe_id), (o.keyframe_id, o.feature_index)
            if pair in seen or feature in features:
                raise ValueError('duplicate observation or feature ownership')
            seen.add(pair); features.add(feature)
            if self.points[o.point_id].host == o.keyframe_id:
                hosts.add(o.point_id)
            # All valid image coordinates are nonnegative: lround(x) = floor(x+0.5).
            pixel = (o.keyframe_id, int(np.floor(o.u + .5)), int(np.floor(o.v + .5)))
            if pixel in samples and samples[pixel] != o.raw_depth:
                raise ValueError('conflicting raw depth samples')
            samples[pixel] = o.raw_depth
            if (pixel[1] >= k.width or pixel[2] >= k.height) and o.raw_depth:
                raise ValueError('depth sample outside image')
        if hosts != set(self.points):
            raise ValueError('missing host observations')
        return self


def load_snapshot(path):
    path = Path(path)
    if path.stat().st_size > 512 * 1024**2:
        raise ValueError('snapshot exceeds input budget')
    tokens = iter(path.read_text().split())
    def take():
        try: return next(tokens)
        except StopIteration as e: raise ValueError('truncated snapshot') from e
    def expect(word):
        if take() != word: raise ValueError('expected ' + word)
    def count(limit):
        n = int(take())
        if not 0 <= n <= limit: raise ValueError('invalid count')
        return n
    expect('AFS_GEOMETRY')
    if take() != '1': raise ValueError('unsupported schema')
    expect('META'); revision, version = take(), int(take())
    expect('CAMERA'); camera = np.array([float(take()) for _ in range(4)])
    expect('CONFIG')
    config = Config(*[int(take()) for _ in range(3)], *[float(take()) for _ in range(3)])
    expect('KEYFRAMES'); keyframes = []
    for _ in range(count(10000)):
        expect('K')
        kid, source, timestamp, width, height = int(take()), int(take()), float(take()), int(take()), int(take())
        keyframes.append(Keyframe(kid, source, timestamp, width, height,
                                  np.array([float(take()) for _ in range(16)]).reshape(4, 4)))
    expect('POINTS'); points = {}
    for _ in range(count(1000000)):
        expect('P'); pid, host = int(take()), int(take())
        if pid in points: raise ValueError('duplicate point ID')
        points[pid] = Point(pid, host, np.array([float(take()) for _ in range(3)]))
    expect('OBSERVATIONS'); observations = []
    for _ in range(count(5000000)):
        expect('O')
        observations.append(Observation(*[int(take()) for _ in range(3)],
                                        float(take()), float(take()), int(take())))
    expect('END')
    if next(tokens, None) is not None: raise ValueError('trailing snapshot data')
    return Snapshot(camera, keyframes, points, observations, config, revision, version).validate()


def write_snapshot(path, s):
    s.validate()
    def numbers(values): return ' '.join(format(float(v), '.17g') for v in values)
    c = s.config
    with Path(path).open('x') as out:
        out.write(f'AFS_GEOMETRY 1\nMETA {s.source_revision} {s.state_version}\nCAMERA {numbers(s.camera)}\n')
        out.write(f'CONFIG {c.window_size} {c.minimum_point_observations} {c.maximum_iterations} '
                  f'{numbers([c.huber_delta_pixels, c.depth_scale, c.depth_residual_weight])}\n')
        out.write(f'KEYFRAMES {len(s.keyframes)}\n')
        for k in s.keyframes:
            out.write(f'K {k.id} {k.source_frame_id} {k.timestamp:.17g} {k.width} {k.height} '
                      f'{numbers(k.t_wc.ravel())}\n')
        out.write(f'POINTS {len(s.points)}\n')
        for p in s.points.values(): out.write(f'P {p.id} {p.host} {numbers(p.xyz)}\n')
        out.write(f'OBSERVATIONS {len(s.observations)}\n')
        # C++ reconstructs the first observation as host; preserve all other order.
        ordered = sorted(enumerate(s.observations), key=lambda x: (
            x[1].point_id, x[1].keyframe_id != s.points[x[1].point_id].host, x[0]))
        for _, o in ordered:
            out.write(f'O {o.point_id} {o.keyframe_id} {o.feature_index} '
                      f'{o.u:.17g} {o.v:.17g} {o.raw_depth}\n')
        out.write('END\n')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_sparse_ply(path,snapshot):
    """Export the snapshot's finite world points, explicitly in metres."""
    snapshot.validate()
    with Path(path).open('x') as out:
        out.write('ply\nformat ascii 1.0\ncomment units metres; optimized shared landmarks\n')
        out.write(f'element vertex {len(snapshot.points)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n')
        for p in snapshot.points.values():out.write(' '.join(f'{x:.17g}' for x in p.xyz)+'\n')
