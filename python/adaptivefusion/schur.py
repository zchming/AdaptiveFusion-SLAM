"""Block Schur LM for the existing RGB-D objective, implemented in Python.

This is a standard BA algorithm, not a novel numerical solver. NumPy supplies
the small dense linear solves. Landmarks are eliminated independently (3x3),
leaving at most 6*(window_size-1) camera variables in the reduced system.
"""
from types import SimpleNamespace
import time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix,diags,issparse
from scipy.sparse.linalg import spsolve
from .geometry import project,skew,so3_right_jacobian


def normal_blocks(problem,x,sparse_camera=None):
    poses,points=problem.unpack(x)
    rotation=Rotation.from_rotvec(poses[problem.ok,:3]).as_matrix()
    pc=np.einsum('nij,nj->ni',rotation,points[problem.op])+poses[problem.ok,3:]
    pixel=project(problem.s.camera,pc)-problem.uv
    fx,fy,_,_=problem.s.camera
    projection=np.zeros((len(pc),3,3))
    projection[:,0,0]=fx/pc[:,2];projection[:,1,1]=fy/pc[:,2]
    projection[:,0,2]=-fx*pc[:,0]/pc[:,2]**2
    projection[:,1,2]=-fy*pc[:,1]/pc[:,2]**2
    projection[:,2,2]=problem.s.config.depth_residual_weight*problem.depth_mask
    depth=(pc[:,2]-problem.depths)*problem.s.config.depth_residual_weight*problem.depth_mask
    residual=np.column_stack([pixel,depth])
    delta=problem.s.config.huber_delta_pixels
    pixel_weight=np.sqrt(np.minimum(1.,delta/np.maximum(np.linalg.norm(pixel,axis=1),1e-30)))
    depth_weight=np.sqrt(np.minimum(1.,delta/np.maximum(np.abs(depth),1e-30)))
    weight=np.column_stack([pixel_weight,pixel_weight,depth_weight])
    angular=-rotation @ skew(points[problem.op]) @ so3_right_jacobian(poses[problem.ok,:3])
    camera=np.concatenate([angular,np.broadcast_to(np.eye(3),angular.shape)],axis=2)
    jp=(projection@rotation)*weight[:,:,None]
    jc=(projection@camera)*weight[:,:,None]
    residual*=weight
    npoint,ncamera=len(problem.pids),problem.npose
    c=np.zeros((npoint,3,3));gc=np.zeros(ncamera);gp=np.zeros((npoint,3))
    np.add.at(c,problem.op,np.einsum('nri,nrj->nij',jp,jp))
    np.add.at(gp,problem.op,np.einsum('nri,nr->ni',jp,residual))
    if sparse_camera is None:sparse_camera=ncamera>60
    if sparse_camera:
        def matrix(rowbase,colbase,array,shape):
            nr,nc=array.shape[1:]
            rows=np.broadcast_to(rowbase[:,None,None]+np.arange(nr)[None,:,None],array.shape)
            cols=np.broadcast_to(colbase[:,None,None]+np.arange(nc)[None,None,:],array.shape)
            return coo_matrix((array.ravel(),(rows.ravel(),cols.ravel())),shape=shape).tocsr()
        free=problem.free_index>=0
        jc_sparse=matrix(3*np.flatnonzero(free),6*problem.free_index[free],jc[free],(3*len(pc),ncamera))
        jp_sparse=matrix(3*np.arange(len(pc)),3*problem.op,jp,(3*len(pc),3*npoint))
        a=(jc_sparse.T@jc_sparse).tocsr();b=(jc_sparse.T@jp_sparse).tocsr()
        gc=np.asarray(jc_sparse.T@residual.ravel()).ravel()
        return a,b,c,gc,gp
    a=np.zeros((ncamera,ncamera));b=np.zeros((npoint,ncamera,3))
    for k in range(len(problem.free)):
        use=problem.free_index==k;sl=slice(k*6,k*6+6)
        a[sl,sl]=np.einsum('nri,nrj->ij',jc[use],jc[use])
        gc[sl]=np.einsum('nri,nr->i',jc[use],residual[use])
        np.add.at(b[:,sl,:],problem.op[use],np.einsum('nri,nrj->nij',jc[use],jp[use]))
    return a,b,c,gc,gp


def schur_step(blocks,damping):
    a,b,c,gc,gp=blocks
    cd=c.copy()
    idx=np.arange(3)
    cd[:,idx,idx]+=damping*np.maximum(c[:,idx,idx],1e-8)
    cinv=np.linalg.inv(cd)
    if issparse(a):
        ad=a+diags(damping*np.maximum(a.diagonal(),1e-8))
        base=3*np.arange(len(cinv))
        rows=np.broadcast_to(base[:,None,None]+np.arange(3)[None,:,None],cinv.shape)
        cols=np.broadcast_to(base[:,None,None]+np.arange(3)[None,None,:],cinv.shape)
        ci=coo_matrix((cinv.ravel(),(rows.ravel(),cols.ravel())),shape=(3*len(cinv),3*len(cinv))).tocsr()
        bci=b@ci
        reduced=ad-bci@b.T
        rhs=-gc+bci@gp.ravel()
        dc=spsolve(((reduced+reduced.T)*.5).tocsc(),rhs)
        dp=-(ci@(gp.ravel()+b.T@dc))
        return np.r_[dc,dp]
    ad=a.copy()
    ad[np.diag_indices_from(ad)]+=damping*np.maximum(np.diag(a),1e-8)
    bcinv=b@cinv
    reduced=ad-np.einsum('nki,nli->kl',bcinv,b)
    rhs=-gc+np.einsum('nki,ni->k',bcinv,gp)
    dc=np.linalg.solve((reduced+reduced.T)*.5,rhs)
    dp=-np.einsum('nij,nj->ni',cinv,gp+np.einsum('nki,k->ni',b,dc))
    return np.r_[dc,dp.ravel()]


def optimize(problem,max_nfev,deadline,ftol,xtol,gtol):
    x=problem.x0.copy();cost=problem.metrics(x)['robust_cost']
    nfev,iterations,accepted=1,0,0
    damping=1e-3
    status,message=0,'maximum number of function evaluations exceeded'
    optimality=float('inf')
    while nfev<max_nfev and iterations<problem.s.config.maximum_iterations:
        if time.perf_counter()>deadline:raise TimeoutError('solver time budget exceeded')
        blocks=normal_blocks(problem,x)
        gradient=np.r_[blocks[3],blocks[4].ravel()]
        optimality=float(np.max(np.abs(gradient)))
        if optimality<gtol:status,message=1,'gradient tolerance reached';break
        iterations+=1
        improved=False
        for _ in range(12):
            if nfev>=max_nfev:break
            if time.perf_counter()>deadline:raise TimeoutError('solver time budget exceeded')
            step=schur_step(blocks,damping)
            if not np.isfinite(step).all():raise ValueError('non-finite Schur step')
            if np.linalg.norm(step)<=xtol*(xtol+np.linalg.norm(x)):
                status,message=3,'step tolerance reached';break
            trial=x+step;nfev+=1
            try:trial_cost=problem.metrics(trial)['robust_cost']
            except ValueError:trial_cost=float('inf')
            if trial_cost<cost:
                difference=cost-trial_cost
                x,cost=trial,trial_cost;damping=max(damping/3,1e-12)
                accepted+=1;improved=True
                if difference<ftol*max(cost,1e-30):status,message=2,'cost tolerance reached'
                break
            damping*=10
        if status:break
        if not improved:status,message=-1,'no acceptable damped step';break
    jac=problem.jacobian(x) if len(x)<=200 else None
    if not status and iterations>=problem.s.config.maximum_iterations:
        message='maximum iterations reached'
    final_blocks=normal_blocks(problem,x)
    optimality=float(np.max(np.abs(np.r_[final_blocks[3],final_blocks[4].ravel()])))
    return SimpleNamespace(x=x,success=status>0,status=status,message=message,nfev=nfev,
                           optimality=optimality,jac=jac,iterations=iterations,accepted_steps=accepted)
