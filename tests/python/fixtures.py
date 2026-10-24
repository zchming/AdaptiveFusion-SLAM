"""Deterministic synthetic geometry. Truth stays outside solver snapshots."""
import numpy as np
from scipy.spatial.transform import Rotation
from adaptivefusion.geometry import inverse, project
from adaptivefusion.state import Snapshot, Keyframe, Point, Observation, Config


def scene(*, noisy=False, exact=False, cameras=3, quantized=True):
    rng = np.random.default_rng(37)
    xyz = np.array([[-.8,-.4,4.],[-.3,.5,4.5],[.2,-.6,5.],
                    [.7,.4,5.5],[-.5,.1,6.],[.6,-.2,6.5],
                    [-.9,.6,4.2],[.8,-.5,5.8]])
    camera = np.array([500.,500.,320.,240.])
    truth, keyframes, observations = [], [], []
    for k in range(cameras):
        pose = np.eye(4)
        pose[:3,:3] = Rotation.from_rotvec(
            [.01*k,.018*k,-.008*k] if quantized else [0,0,-.008*k]).as_matrix()
        pose[:3,3] = [.12*k,.015*k,.01*k]
        truth.append(pose)
        initial = pose.copy()
        if k and not exact:
            initial[:3,:3] = Rotation.from_rotvec([.008,-.012,.006]).as_matrix() @ pose[:3,:3]
            initial[:3,3] += [.04,-.025,.018]
        keyframes.append(Keyframe(k, 2*k, .1+float(k),640,480,initial))
        pc = (inverse(pose) @ np.c_[xyz,np.ones(len(xyz))].T).T[:,:3]
        uv = project(camera,pc)
        if noisy: uv += rng.normal(0,.25,uv.shape)
        # Match the C++ keypoint's float storage; raw depth is uint16.
        uv = uv.astype(np.float32).astype(float)
        depth = np.rint(pc[:,2]*5000).astype(int)
        if noisy: depth += rng.integers(-15,16,len(depth))
        for p,(pixel,z) in enumerate(zip(uv,depth)):
            observations.append(Observation(100+3*p,k,p,*pixel,int(z)))
    points = {100+3*p:Point(100+3*p,0,x.copy() if exact else x+rng.normal(0,.035,3))
              for p,x in enumerate(xyz)}
    snapshot = Snapshot(camera,keyframes,points,observations,Config(maximum_iterations=100),
                        'synthetic',42).validate()
    return snapshot, truth, xyz
