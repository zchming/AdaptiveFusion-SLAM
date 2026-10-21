"""Strict reader for original C++ ORB rows, plus explicit Python match policy."""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import shlex
import numpy as np


@dataclass
class FrontendFeatures:
    metadata: dict
    keypoints: np.ndarray  # u,v,size,angle,response,octave,class_id
    raw_depth: np.ndarray
    point_ids: np.ndarray
    descriptors: np.ndarray


def read_features(path):
    path=Path(path)
    if path.stat().st_size>30*1024**2:raise ValueError('feature sidecar budget exceeded')
    lines=path.read_text().splitlines()
    if len(lines)<10 or lines[0]!='AFS_FEATURES 1' or lines[-1]!='END':raise ValueError('invalid/truncated feature schema')
    def row(i,tag,size):
        values=shlex.split(lines[i])
        if len(values)!=size or values[0]!=tag:raise ValueError('invalid feature '+tag)
        return values[1:]
    meta=row(1,'META',4);key=row(2,'KEYFRAME',7)
    camera=np.array(row(3,'CAMERA',5),float);depth=np.array(row(4,'DEPTH',4),float)
    orb=row(5,'ORB',5);match=row(6,'MATCH',4);images=row(7,'IMAGES',3);count=row(8,'FEATURES',3)
    source,version,phase=meta;n,width=int(count[0]),int(count[1])
    kid,fid=int(key[0]),int(key[1]);rgb_ts,depth_ts=map(float,key[2:4]);w,h=map(int,key[4:])
    max_features,levels,fast=int(orb[0]),int(orb[2]),int(orb[3]);scale=float(orb[1])
    ratio,max_distance,mutual=float(match[0]),int(match[1]),int(match[2])
    if (phase!='after_insertion_before_ba' or not source or int(version)!=fid or min(kid,fid)<0
            or width!=32 or not 0<=n<=100000 or len(lines)!=n+10 or min(w,h)<=0 or max(w,h)>4096
            or not np.isfinite([rgb_ts,depth_ts,*camera,*depth,scale,ratio]).all()
            or abs(rgb_ts-depth_ts)>.02 or min(camera[:2])<=0
            or not 0<depth[1]<depth[2] or depth[0]<=0 or max_features<=0 or levels<=0 or fast<0
            or scale<=1 or not 0<ratio<1 or not 0<=max_distance<=256 or mutual not in [0,1]):
        raise ValueError('invalid native feature metadata')
    for relative in images:
        p=PurePosixPath(relative)
        if p.is_absolute() or '..' in p.parts or not p.parts or '\\' in relative or any(ord(c)<32 for c in relative):
            raise ValueError('unsafe relative image reference')
    kp=np.empty((n,7),float);raw=np.empty(n,np.uint16);ids=np.empty(n,np.int64);descriptors=np.empty((n,32),np.uint8)
    for i in range(n):
        r=lines[i+9].split()
        if len(r)!=12 or r[0]!='F' or int(r[1])!=i:raise ValueError('feature index/schema mismatch')
        values=np.array(r[2:7],float);octave,category=int(r[7]),int(r[8]);sample,pid=int(r[9]),int(r[10])
        if (not np.isfinite(values).all() or not np.array_equal(values,values.astype(np.float32).astype(float))
                or not 0<=values[0]<w or not 0<=values[1]<h or values[2]<=0 or not -1<=values[3]<360
                or not 0<=octave<levels or not -(2**31)<=category<2**31
                or not 0<=sample<=65535 or not -1<=pid<2**63 or len(r[11])!=64
                or any(c not in '0123456789abcdef' for c in r[11])):
            raise ValueError('invalid native feature row')
        x,y=np.floor(values[:2]+.5).astype(int)
        if (x>=w or y>=h) and sample:raise ValueError('depth outside image')
        kp[i]=[*values,octave,category];raw[i]=sample;ids[i]=pid
        descriptors[i]=np.frombuffer(bytes.fromhex(r[11]),np.uint8)
    mapped=ids[ids>=0]
    if len(set(mapped.tolist()))!=len(mapped):raise ValueError('duplicate feature map ownership')
    metadata=dict(source_revision=source,state_version=int(version),capture_phase=phase,keyframe_id=kid,
                  source_frame_id=fid,rgb_timestamp=rgb_ts,depth_timestamp=depth_ts,width=w,height=h,
                  camera=camera.tolist(),depth=depth.tolist(),orb=[max_features,scale,levels,fast],
                  matching=[ratio,max_distance,bool(mutual)],images=images)
    return FrontendFeatures(metadata,kp,raw,ids,descriptors)


_BITS=np.array([int(i).bit_count() for i in range(256)],np.uint8)


def descriptor_matches(query, train, ratio=.75, maximum_distance=64, mutual=True):
    """NumPy Hamming/ratio/mutual policy matching C++ defaults, in bounded blocks.

    Returns feature-row IDs, not a claim to replay a frontend decision. Tie
    ordering uses first row IDs; forward equal-best distances fail the ratio.
    """
    a,b=np.asarray(query),np.asarray(train)
    if any(x.dtype!=np.uint8 or x.ndim!=2 or x.shape[1]!=32 for x in [a,b]):
        raise ValueError('native uint8 Nx32 descriptors required')
    if not 0<ratio<1 or not 0<=maximum_distance<=256:raise ValueError('invalid descriptor policy')
    if not len(a) or len(b)<2:return []
    forward=[];reverse_distance=np.full(len(b),257,np.int16);reverse_index=np.full(len(b),-1,int)
    for start in range(0,len(a),64):
        distances=_BITS[np.bitwise_xor(a[start:start+64,None,:],b[None,:,:])].sum(axis=2,dtype=np.uint16)
        for offset,row in enumerate(distances):
            i=start+offset;j=int(np.argmin(row));best=int(row[j]);second=int(np.partition(row,1)[1])
            if best<=maximum_distance and best<ratio*second:forward.append((i,j,best))
            improved=row<reverse_distance;reverse_distance[improved]=row[improved];reverse_index[improved]=i
    return [(i,j,d) for i,j,d in forward if not mutual or reverse_index[j]==i]
