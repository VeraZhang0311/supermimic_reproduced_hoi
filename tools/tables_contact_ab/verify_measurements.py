"""Independent max-depth checks: all-pair query and Minkowski hull geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def triangle_distance(a,b,c):
    # Closest point on a triangle to the origin, including edge/vertex regions.
    ab,ac=b-a,c-a;ap=-a;d1,d2=ab@ap,ac@ap
    if d1<=0 and d2<=0:return np.linalg.norm(a)
    bp=-b;d3,d4=ab@bp,ac@bp
    if d3>=0 and d4<=d3:return np.linalg.norm(b)
    vc=d1*d4-d3*d2
    if vc<=0 and d1>=0 and d3<=0:return np.linalg.norm(a+(d1/(d1-d3))*ab)
    cp=-c;d5,d6=ab@cp,ac@cp
    if d6>=0 and d5<=d6:return np.linalg.norm(c)
    vb=d5*d2-d1*d6
    if vb<=0 and d2>=0 and d6<=0:return np.linalg.norm(a+(d2/(d2-d6))*ac)
    va=d3*d6-d5*d4
    if va<=0 and d4-d3>=0 and d5-d6>=0:return np.linalg.norm(b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b))
    denom=1/(va+vb+vc)
    return np.linalg.norm(a+ab*(vb*denom)+ac*(vc*denom))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    result=[];R=lambda q:Rotation.from_quat(np.asarray(q)[...,[1,2,3,0]]).as_matrix()
    for variant in ('A','B'):
        root=args.root/variant
        g=json.loads((root/'geometry.json').read_text())['geoms']
        saved=dict(np.load(root/'substeps.npz',allow_pickle=False));depth=np.load(root/'penetration.npz')['depth_m']
        names=list(map(str,saved['names']));m=mujoco.MjModel.from_binary_path(str(root/'native_model.mjb'));d=mujoco.MjData(m)
        frame=int(depth.max(1).argmax())
        for geom in g:
            role=geom['role']
            if role==2:p,rot=saved['actual_object_pos'][frame],R(saved['actual_object_quat'][frame])
            else:
                i=names.index('left_ankle_roll_link' if role==3 else 'right_ankle_roll_link')
                p,rot=saved['actual_pos'][frame,i],R(saved['actual_quat'][frame,i])
            d.geom_xpos[geom['id']]=p+rot@np.asarray(geom['geom_pos'])
            d.geom_xmat[geom['id']]=(rot@R(geom['geom_quat'])).reshape(9)
        best=(0.,None,None)
        for foot in [x for x in g if x['role'] in (3,4)]:
            for obj in [x for x in g if x['role']==2]:
                distance=mujoco.mj_geomDistance(m,d,foot['id'],obj['id'],1.,None)
                if -distance>best[0]:best=(-distance,foot,obj)
        assert abs(best[0]-float(depth[frame].max()))<1e-8
        row={'variant':variant,'maximum_frame_200hz':frame,'time_s':frame*.005,'all_pair_maximum_mm':best[0]*1000,
             'bounding_sphere_pruning_checked':True}
        if best[0]>0:
            _,foot,obj=best;mid=obj['geom_dataid'];adr,n=m.mesh_vertadr[mid],m.mesh_vertnum[mid]
            vertices=m.mesh_vert[adr:adr+n]@d.geom_xmat[obj['id']].reshape(3,3).T+d.geom_xpos[obj['id']]
            center=d.geom_xpos[foot['id']];axis=d.geom_xmat[foot['id']].reshape(3,3)@np.array([0.,0.,1.])
            radius,half=foot['geom_size'][:2]
            points=np.concatenate((vertices-center-half*axis,vertices-center+half*axis))
            hull=ConvexHull(points);offset=float(hull.equations[:,3].max())
            if offset<=0:signed=offset-radius
            else:signed=min(triangle_distance(*points[f]) for f in hull.simplices)-radius
            error=abs(-signed-best[0]);assert error<2e-5
            row.update(independent_minkowski_depth_mm=-signed*1000,independent_error_mm=error*1000)
        result.append(row)
    output=args.root/'independent_geometry_verification.json'
    with output.open('x') as stream:json.dump({'passed':True,'checks':result,
        'verification_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},stream,indent=2)
    print(json.dumps({'passed':True,'checks':result},indent=2))


if __name__=='__main__':main()
