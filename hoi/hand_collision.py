"""Original HOI rubber-hand meshes added as independent, explicitly cooked colliders."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET

HAND_SHA = {'left': 'cff2221a690fa69303f61fce68f2d155c1517b52efb6ca9262dd56e0bc6e70fe',
            'right': '99533b778bca6246144fa511bb9d4e555e075c641f2a0251e04372869cd99d67'}
ORIGIN = {'left': [.0415, .003, 0.], 'right': [.0415, -.003, 0.]}
RECIPE = dict(max_convex_hull=5, max_ch_vertex=16, decimate=True, merge=True,
              mcts_nodes=20, mcts_iterations=5, mcts_max_depth=1, seed=0)
LIMITATION = ('Original HOI requested PhysX convexDecomposition (5 hulls, 16 vertices, '
    '60000 voxel resolution); no cooked vertices were saved. This comparison reuses '
    'the byte-identical STL and mounting transform, cooks only the two hands with '
    'installed CoACD (5 hull / 16 vertex limits), then imports explicit convex hulls. '
    'CoACD is not the original PhysX cooker; its sampling resolution is not a mapping '
    'of PhysX voxel resolution. Objects and all existing robot colliders are unchanged. '
    'Contact/rest offsets were automatic in original HOI and remain current-backend defaults.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_definition(path, require_collision=True):
    root = ET.parse(path).getroot()
    out = {}
    for side in HAND_SHA:
        name = side + '_rubber_hand'
        link = root.find(f"link[@name='{name}']")
        collision = link.findall('collision')
        assert (len(collision) == 1 if require_collision else len(collision) in (0, 1))
        geometry = collision[0] if collision else link.find('visual')
        mesh = geometry.find('geometry/mesh')
        assert Path(mesh.get('filename')).name == name + '.STL'
        assert [float(x) for x in mesh.get('scale', '1 1 1').split()] == [1., 1., 1.]
        origin = geometry.find('origin')
        assert [float(x) for x in origin.get('xyz', '0 0 0').split()] == [0., 0., 0.]
        assert [float(x) for x in origin.get('rpy', '0 0 0').split()] == [0., 0., 0.]
        joint = next(j for j in root.findall('joint') if j.find('child').get('link') == name)
        assert joint.get('type') == 'fixed'
        assert joint.find('parent').get('link') == side + '_wrist_yaw_link'
        mount = joint.find('origin')
        xyz = [float(x) for x in mount.get('xyz').split()]
        assert xyz == ORIGIN[side]
        assert [float(x) for x in mount.get('rpy', '0 0 0').split()] == [0., 0., 0.]
        out[side] = {'stl_filename': mesh.get('filename'), 'fixed_joint_origin': xyz,
                     'collision_origin': [0., 0., 0.], 'mesh_scale': [1., 1., 1.]}
    return out


def stage_snapshot(stage):
    """Authored values/relations of every original prim, including instance proxies."""
    from pxr import Usd
    return {str(p.GetPath()): {
        'attributes': {a.GetName(): str(a.Get()) for a in p.GetAttributes() if a.HasAuthoredValueOpinion()},
        'relationships': {r.GetName(): [str(x) for x in r.GetTargets()] for r in p.GetRelationships()}}
        for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())}


def prepare(root, output, original_urdf=None):
    # CoACD/trimesh followed by pxr.Tf segfaults in this pinned runtime. Keep the
    # mesh cooker in a child interpreter; USD authoring never loads those libs.
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
    import numpy as np
    root, output = Path(root), Path(output)
    output.mkdir(exist_ok=False)
    definition_source = original_urdf or root / 'data/g1_description/g1_29dof_rev_1_0.urdf'
    definitions = original_definition(definition_source, require_collision=original_urdf is not None)
    source = root / 'data/g1_usd/g1_29dof/g1_29dof.usda'
    src = Usd.Stage.Open(str(source))
    session = Sdf.Layer.CreateAnonymous()
    src = Usd.Stage.Open(src.GetRootLayer(), session)
    with Usd.EditContext(src, session):
        src.GetDefaultPrim().GetVariantSet('Physics').SetVariantSelection('mujoco')
    before = stage_snapshot(src)
    target = output / 'g1_rubber_hand_collision.usda'
    layer = Sdf.Layer.CreateNew(str(target))
    layer.subLayerPaths = [str(source.resolve())]
    stage = Usd.Stage.Open(layer)
    root_prim = stage.GetPrimAtPath(src.GetDefaultPrim().GetPath())
    stage.SetDefaultPrim(root_prim)
    root_prim.GetVariantSet('Physics').SetVariantSelection('mujoco')
    hand_info = {}
    for side in HAND_SHA:
        stl = root / 'data/g1_description/meshes' / (side + '_rubber_hand.STL')
        assert sha(stl) == HAND_SHA[side], stl
        body = next(p for p in stage.Traverse() if p.GetName() == side + '_rubber_hand'
                    and p.HasAPI(UsdPhysics.RigidBodyAPI))
        # Collider points are in the existing fixed hand link's local frame. No
        # rigid body, inertial, joint, PD, visual or wrist geometry is changed.
        translation = list(UsdGeom.Xformable(body).GetLocalTransformation().ExtractTranslation())
        assert np.allclose(translation, ORIGIN[side], atol=1e-7), translation
        cooked=output/(side+'-hulls.json')
        with (output/(side+'-cook.log')).open('x') as log:
            subprocess.run([sys.executable,'-X','faulthandler','-m','hoi.hand_collision',
                '--cook',side,'--stl',str(stl),'--output',str(cooked)],
                check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        pieces=[(np.asarray(p['vertices']),np.asarray(p['faces'])) for p in json.loads(cooked.read_text())['pieces']]
        assert 1 <= len(pieces) <= 5, len(pieces)
        parts = []
        for index, (vertices, faces) in enumerate(pieces):
            assert 4 <= len(vertices) <= 16, len(vertices)
            assert np.isfinite(vertices).all()
            path = body.GetPath().AppendChild(f'HOIRubberCollision{index}')
            collider = UsdGeom.Mesh.Define(stage, path)
            collider.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in vertices])
            collider.CreateFaceVertexCountsAttr([3] * len(faces))
            collider.CreateFaceVertexIndicesAttr(np.asarray(faces).reshape(-1).tolist())
            collider.CreateSubdivisionSchemeAttr('none')
            collider.CreatePurposeAttr('guide')
            UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr('convexHull')
            parts.append({'path': str(path), 'vertices': len(vertices), 'triangles': len(faces)})
        hand_info[side] = {**definitions[side], 'stl_sha256': sha(stl), 'parts': parts}
    after = stage_snapshot(stage)
    assert all(after[path] == values for path, values in before.items()), 'An original authored property changed'
    added = set(after) - set(before)
    assert len(added) == sum(len(h['parts']) for h in hand_info.values())
    assert all('HOIRubberCollision' in p for p in added)
    layer.Save()
    report = {'source_usd': str(source.resolve()), 'source_usd_sha256': sha(source),
        'original_hoi_urdf_sha256': sha(original_urdf) if original_urdf else None,
        'hand_definition_sha256': sha(definition_source), 'recipe': RECIPE,
        'cooker': 'CoACD', 'cooker_and_usd_imports_isolated':True, 'limitations': LIMITATION, 'hands': hand_info,
        'original_authored_values_unchanged': True, 'added_collider_count': len(added),
        'usd': str(target), 'usd_sha256': sha(target)}
    (output / 'hand-asset.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def compiled_audit(model, enabled):
    """Compiled shape masks plus isolated CPU contact probes; never mutate GPU state."""
    import mujoco
    import numpy as np
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or '' for i in range(model.ngeom)]
    bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or '' for i in range(model.nbody)]
    objects = {i for i, name in enumerate(bodies) if '_Object' in name or '/Object' in name}
    object_geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in objects]
    excluded = set(map(int, model.exclude_signature))
    def allowed(a, b):
        ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
        return ((min(ba, bb) << 16) + max(ba, bb)) not in excluded and bool(
            (int(model.geom_contype[a]) & int(model.geom_conaffinity[b])) |
            (int(model.geom_contype[b]) & int(model.geom_conaffinity[a])))
    report = {'enabled': enabled, 'hands': {}}
    for side in HAND_SHA:
        indices = [i for i, name in enumerate(names) if side + '_rubber_hand' in name
                   and (int(model.geom_contype[i]) or int(model.geom_conaffinity[i]))]
        pairs = [(h, o) for h in indices for o in object_geoms if allowed(h, o)]
        assert bool(indices) == bool(enabled), (side, indices, names)
        assert bool(pairs) == bool(enabled), (side, pairs)
        contacts = []
        if enabled:
            for h, o in pairs:
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                body = int(model.geom_bodyid[o])
                joint = int(model.body_jntadr[body])
                assert int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE)
                address = int(model.jnt_qposadr[joint])
                # Place this object hull over the hand hull in an independent
                # CPU Data object. Original simulation state remains untouched.
                delta = data.geom_xpos[h] - data.geom_xpos[o]
                data.qpos[address:address + 3] += delta
                mujoco.mj_forward(model, data)
                found = [c for c in data.contact[:data.ncon] if
                         (int(c.geom1) in indices and int(c.geom2) in object_geoms) or
                         (int(c.geom2) in indices and int(c.geom1) in object_geoms)]
                if found:
                    contacts = [{'geom1': names[int(c.geom1)], 'geom2': names[int(c.geom2)],
                                 'distance': float(c.dist)} for c in found]
                    break
            assert contacts, (side, 'Enabled masks but CPU contact probe found no hand/object contacts')
        report['hands'][side] = {'collision_geom_count': len(indices),
            'allowed_object_pairs': len(pairs), 'geom_names': [names[i] for i in indices],
            'cpu_probe_contacts': contacts}
    # Only robot bodies/joints are compared between OFF and ON: the input object
    # and all old collider source opinions are already frozen/hashed separately.
    report['robot_bodies'] = {name: {'mass': float(model.body_mass[i]),
        'inertia': model.body_inertia[i].tolist(), 'ipos': model.body_ipos[i].tolist(),
        'iquat': model.body_iquat[i].tolist(), 'pos': model.body_pos[i].tolist(),
        'quat': model.body_quat[i].tolist()}
        for i, name in enumerate(bodies) if i and i not in objects}
    report['joints'] = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): {
        'type': int(model.jnt_type[i]), 'range': model.jnt_range[i].tolist(),
        'axis': model.jnt_axis[i].tolist(), 'pos': model.jnt_pos[i].tolist()}
        for i in range(model.njnt) if int(model.jnt_bodyid[i]) not in objects}
    return report


def assert_same_physics(off, on):
    import numpy as np
    for group in ('robot_bodies', 'joints'):
        assert off[group].keys() == on[group].keys(), group
        for name in off[group]:
            for field in off[group][name]:
                assert np.allclose(off[group][name][field], on[group][name][field],
                    rtol=2e-6, atol=2e-7), (group, name, field)
    assert not off['enabled'] and on['enabled']


def cook(side,stl,output):
    import coacd
    import numpy as np
    import trimesh
    assert sha(stl)==HAND_SHA[side]
    coacd.set_log_level('warn')
    mesh=trimesh.load_mesh(str(stl),process=True)
    pieces=coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices),np.asarray(mesh.faces)),**RECIPE)
    assert 1<=len(pieces)<=5
    assert all(4<=len(v)<=16 and np.isfinite(v).all() for v,f in pieces)
    Path(output).write_text(json.dumps({'recipe':RECIPE,'stl_sha256':sha(stl),
        'pieces':[{'vertices':v.tolist(),'faces':f.tolist()} for v,f in pieces]},allow_nan=False)+'\n')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--cook',choices=['left','right'],required=True)
    parser.add_argument('--stl',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    arguments=parser.parse_args()
    cook(arguments.cook,arguments.stl,arguments.output)
