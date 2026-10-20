"""Create a new synthetic exchange fixture, never download or read real data."""
import argparse
import json
from pathlib import Path
from fixtures import scene
from adaptivefusion.state import write_snapshot

p = argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir()
s,poses,xyz=scene()
write_snapshot(a.output/'input.snapshot',s)
(a.output/'truth.json').write_text(json.dumps({'poses':[x.tolist() for x in poses], 'points':xyz.tolist()}))
with (a.output/'frames.txt').open('x') as f:
    f.write('frame_id timestamp valid reference_id anchor_id '+' '.join(f't{i}' for i in range(16))+'\n')
    for k,key in enumerate(s.keyframes):
        f.write(f'{2*k} {key.timestamp:.17g} 1 {2*k-1 if k else -1} {k} '
                +' '.join(format(x,'.17g') for x in key.t_wc.ravel())+'\n')
        follow=key.t_wc.copy();follow[0,3]+=.03
        f.write(f'{2*k+1} {key.timestamp+.25:.17g} 1 {2*k} {k} '
                +' '.join(format(x,'.17g') for x in follow.ravel())+'\n')
