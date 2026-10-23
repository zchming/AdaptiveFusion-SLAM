"""Strict file bridge to the unchanged C++ PnpPoseEstimator; no image/GT input."""
from pathlib import Path
import json
import subprocess
import time
import numpy as np
from .geometry import check_pose


def write_pair(path, source, pixels, camera, ids, seed):
    source, pixels, camera = (np.asarray(x, float) for x in (source, pixels, camera))
    ids = np.asarray(ids)
    if (source.ndim != 2 or source.shape[1:] != (3,)
            or pixels.shape != (len(source), 2) or camera.shape != (4,)
            or ids.shape != (len(source),) or not np.issubdtype(ids.dtype, np.integer)
            or len(set(ids.tolist())) != len(ids) or np.any(ids < 0)
            or np.any(ids > 2**31-1) or len(source) > 100000
            or not all(np.isfinite(x).all() for x in (source, pixels, camera))
            or min(camera[:2]) <= 0 or np.any(source[:, 2] <= 0)
            or type(seed) is not int or not 0 <= seed <= 2**31-1):
        raise ValueError('invalid native PnP input')
    with Path(path).open('x') as f:
        f.write('AFS_PNP_PAIR 1\nCAMERA '+' '.join(format(x, '.17g') for x in camera))
        f.write(f'\nSEED {seed}\nCORRESPONDENCES {len(source)}\n')
        for i, point, uv in zip(ids, source, pixels):
            f.write(f'C {i} '+' '.join(format(x, '.17g') for x in [*point, *uv])+'\n')
        f.write('END\n')


def native_fit(binary, source, pixels, camera, ids, initial, seed, output):
    """Preserve failed processes and return an exact raw fallback, never identity."""
    initial = check_pose(initial).copy()
    output = Path(output)
    output.mkdir()  # A unique arm directory is mandatory, including failed arms.
    input_path = output/'input.pnp'
    write_pair(input_path, source, pixels, camera, ids, seed)
    result = dict(applied=False, t_target_source=initial.tolist(), seed=seed,
                  reason='native process failed', process_returncode=None, timed_out=False)
    begin = time.perf_counter()
    try:
        run = subprocess.run([str(Path(binary).resolve()), '--input', str(input_path),
                              '--output', str(output/'native')],
                             capture_output=True, text=True, timeout=60)
        (output/'process.log').write_text(run.stdout+run.stderr)
        result['process_returncode'] = run.returncode
        if run.returncode == 0:
            native = json.loads((output/'native/report.json').read_text())
            if (native['schema'] != 1 or not native['opencv_version'].startswith('4.6.')
                    or type(native['applied']) is not bool or native['seed'] != seed
                    or native['input_count'] != len(source)
                    or len(set(native['inlier_source_ids'])) != len(native['inlier_source_ids'])
                    or not set(native['inlier_source_ids']) <= set(map(int, ids))):
                raise ValueError('invalid native report/version/identity')
            result.update(native)
            if native['applied']:
                transform = check_pose(np.array(native['t_target_source'], float))
                if len(native['inlier_source_ids']) < 6:
                    raise ValueError('successful native report has insufficient inliers')
                result['t_target_source'] = transform.tolist()
            else:
                result['t_target_source'] = initial.tolist()
    except subprocess.TimeoutExpired as error:
        result.update(timed_out=True, reason='native timeout; exact raw fallback')
        (output/'process.log').write_text(str(error))
    except (ValueError, KeyError, OSError, TypeError) as error:
        result.update(applied=False, reason=str(error), t_target_source=initial.tolist())
    result['wall_s'] = time.perf_counter()-begin
    (output/'fit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result
