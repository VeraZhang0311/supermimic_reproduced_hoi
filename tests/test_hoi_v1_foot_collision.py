"""CPU provenance/diff contracts; no PhysX or GPU cooking pass is inferred."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from hoi.hand_collision import HAND_SHA, RECIPE as HAND_RECIPE
from hoi.v1 import foot_collision as fc


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def record(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def urdf_text():
    return '<robot name="G1">' + ''.join(f'''
      <link name="{side}_ankle_roll_link"><visual><geometry><mesh filename="visual.STL"/></geometry></visual>
        <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry>
          <mesh filename="meshes/{side}_ankle_roll_link.STL"/>
        </geometry></collision></link>
      <joint name="{side}_ankle_roll_joint" type="revolute"><parent link="{side}_ankle_pitch_link"/>
        <child link="{side}_ankle_roll_link"/><origin xyz="0 0 -0.017558" rpy="0 0 0"/>
      </joint>''' for side in ("left", "right")) + '</robot>'


def hull():
    return {"vertices": [[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.], [0., 0., .1]],
            "faces": [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]}


def stage_fixture():
    before = {"/G1/torso": {"type": "Xform", "schemas": ["PhysicsMassAPI"],
                           "attributes": {"physics:mass": "10", "drive:stiffness": "40"}, "relationships": {}}}
    disabled, added = [], []
    for side in ("left", "right"):
        for index in range(4):
            path = f"/G1/{side}_ankle_roll_link/Sphere{index}"
            before[path] = {"type": "Sphere", "schemas": ["PhysicsCollisionAPI"],
                            "attributes": {"radius": "0.005", "physics:collisionEnabled": "True"}, "relationships": {}}
            disabled.append(path)
        added.append(f"/G1/{side}_ankle_roll_link/HOIUltraFootCollision0")
    after = deepcopy(before)
    for path in disabled:
        after[path]["attributes"]["physics:collisionEnabled"] = "False"
    for path in added:
        after[path] = {"type": "Mesh", "schemas": ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"],
                       "attributes": {"physics:collisionEnabled": "True", "physics:approximation": "convexHull"}, "relationships": {}}
    return before, after, disabled, added


class FootCollisionTests(unittest.TestCase):
    def test_versioned_sources_and_cook_limit_are_explicitly_not_physx_equivalence(self):
        self.assertEqual(fc.ULTRA_URDF_SHA256, "1791b76b498fd60f4b29dfbf6cc82d2144df6ea739ef6d3cf32442168a3097a1")
        self.assertEqual(fc.FOOT_SHA["left"], "c4092af943141d4d9f74232f3cfa345afc6565f46a077793b8ae0e68b39dc33f")
        self.assertEqual(fc.FOOT_SHA["right"], "4b66222ea56653e627711b56d0a8949b4920da5df091da0ceb343f54e884e3a5")
        self.assertEqual(fc.RECIPE, HAND_RECIPE)
        self.assertEqual((fc.RECIPE["max_convex_hull"], fc.RECIPE["max_ch_vertex"]), (5, 16))
        self.assertNotIn("resolution", fc.RECIPE)
        self.assertEqual(fc.ORIGINAL_VHACD["resolution"], 60000)
        self.assertIn("does not reproduce", fc.LIMITATION)

    def test_collision_mesh_scale_origin_and_joint_mount_are_read_from_urdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'g1.urdf'
            path.write_text(urdf_text())
            definitions = fc.original_definition(path)
            for side in ("left", "right"):
                self.assertEqual(definitions[side]["link"], side + "_ankle_roll_link")
                self.assertEqual(definitions[side]["mesh_scale"], [1., 1., 1.])
                self.assertEqual(definitions[side]["collision_origin"], [0., 0., 0.])
                self.assertEqual(definitions[side]["joint_origin"], [0., 0., -.017558])
            for mutation in ('visual_only', 'sphere', 'scale', 'origin', 'mount'):
                root = ET.fromstring(urdf_text())
                link = root.find("link[@name='left_ankle_roll_link']")
                if mutation == 'visual_only':
                    link.remove(link.find('collision'))
                elif mutation == 'sphere':
                    geometry = link.find('collision/geometry')
                    geometry.remove(geometry.find('mesh'))
                    ET.SubElement(geometry, 'sphere', radius='.005')
                elif mutation == 'scale':
                    link.find('collision/geometry/mesh').set('scale', '1 1 .99')
                elif mutation == 'origin':
                    link.find('collision/origin').set('xyz', '0 0 .001')
                else:
                    root.find('joint/origin').set('xyz', '0 0 -.017')
                ET.ElementTree(root).write(path)
                with self.assertRaises(ValueError, msg=mutation):
                    fc.original_definition(path)

    def test_mesh_resolution_supports_relocated_explicit_paths_and_local_uris(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            urdf = folder / 'g1.urdf'
            urdf.write_text(urdf_text())
            mesh = folder / 'meshes/left_ankle_roll_link.STL'
            mesh.parent.mkdir()
            mesh.write_bytes(b'source STL')
            self.assertEqual(fc.resolve_mesh_uri(urdf, 'meshes/left_ankle_roll_link.STL'), mesh.resolve())
            self.assertEqual(fc.resolve_mesh_uri(urdf, mesh.as_uri()), mesh.resolve())
            self.assertEqual(fc.resolve_mesh_uri(urdf, 'package://G1/meshes/left_ankle_roll_link.STL'), mesh.resolve())
            self.assertEqual(fc.resolve_mesh_uri(urdf, 'missing.STL', mesh), mesh.resolve())
            with self.assertRaises(ValueError):
                fc.resolve_mesh_uri(urdf, 'https://example.invalid/foot.STL')

    def test_hull_validation_rejects_wrong_source_recipe_nan_limits_and_open_faces(self):
        good = {"recipe": fc.RECIPE, "stl_sha256": fc.FOOT_SHA['left'], "pieces": [hull()]}
        self.assertEqual(len(fc.validate_hulls(good, 'left')), 1)
        for mutation in ('source', 'recipe', 'six_hulls', 'vertices', 'nan', 'open', 'face_index', 'degenerate'):
            bad = deepcopy(good)
            if mutation == 'source': bad['stl_sha256'] = '0' * 64
            elif mutation == 'recipe': bad['recipe']['max_ch_vertex'] = 17
            elif mutation == 'six_hulls': bad['pieces'] = [hull()] * 6
            elif mutation == 'vertices': bad['pieces'][0]['vertices'] *= 5
            elif mutation == 'nan': bad['pieces'][0]['vertices'][0][0] = float('nan')
            elif mutation == 'open': bad['pieces'][0]['faces'].pop()
            elif mutation == 'degenerate': bad['pieces'][0]['vertices'] = [[0., 0., 0.]] * 4
            else: bad['pieces'][0]['faces'][0][0] = 99
            with self.assertRaises(ValueError, msg=mutation):
                fc.validate_hulls(bad, 'left')

    def test_snapshot_diff_preserves_mass_pd_visual_joints_and_all_old_prims(self):
        before, after, disabled, added = stage_fixture()
        self.assertTrue(fc.verify_stage_difference(before, after, disabled, added)['passed'])
        for mutation in ('mass', 'pd', 'delete', 'visual', 'new_body', 'new_mesh_mass', 'nonfoot'):
            bad = deepcopy(after)
            if mutation == 'mass': bad['/G1/torso']['attributes']['physics:mass'] = '11'
            elif mutation == 'pd': bad['/G1/torso']['attributes']['drive:stiffness'] = '41'
            elif mutation == 'delete': del bad['/G1/torso']
            elif mutation == 'visual': bad['/G1/torso']['attributes']['visibility'] = 'invisible'
            elif mutation == 'new_body': bad['/G1/new'] = deepcopy(bad[added[0]])
            elif mutation == 'new_mesh_mass': bad[added[0]]['schemas'].append('PhysicsMassAPI')
            else:
                disabled = disabled + ['/G1/torso']
                bad['/G1/torso']['attributes']['physics:collisionEnabled'] = 'False'
            with self.assertRaises(ValueError, msg=mutation):
                fc.verify_stage_difference(before, bad, disabled, added)

    def fixture(self, folder):
        """CPU schema fixture; USD bytes are opaque, never native-pass evidence."""
        base_dir, out = folder / 'base', folder / 'overlay'
        base_dir.mkdir(); out.mkdir()
        source, base, target = base_dir / 'source.usda', base_dir / 'hands.usda', out / 'feet.usda'
        source.write_bytes(b'original USD'); base.write_bytes(b'bound hand overlay'); target.write_bytes(b'new foot overlay')
        hand = {"source_usd": str(source), "source_usd_sha256": fc.sha(source), "usd": str(base), "usd_sha256": fc.sha(base),
                "recipe": dict(HAND_RECIPE), "original_authored_values_unchanged": True,
                "hands": {side: {"stl_sha256": value} for side, value in HAND_SHA.items()}}
        original_hand = base_dir / 'hand-asset.json'
        write(original_hand, hand)
        (out / 'hand-asset.json').write_bytes(original_hand.read_bytes())
        urdf = folder / 'g1.urdf'; urdf.write_text(urdf_text())
        definitions = fc.original_definition(urdf)
        meshes, expected = {}, {}
        for side in ('left', 'right'):
            path = folder / (side + '.STL'); path.write_bytes((side + ' test source').encode())
            expected[side] = fc.sha(path)
            meshes[side] = {**definitions[side], **record(path)}
        before, after, disabled, added = stage_fixture()
        sides = {}
        for side in ('left', 'right'):
            cooked = out / (side + '-hulls.json')
            write(cooked, {"recipe": fc.RECIPE, "stl_sha256": expected[side], "pieces": [hull()]})
            sides[side] = {"link": definitions[side]['link'], "hull_count": 1, "cooked_geometry": record(cooked),
                "disabled_colliders": [p for p in disabled if f'/{side}_' in p],
                "parts": [{"path": next(p for p in added if f'/{side}_' in p), "vertices": 4, "triangles": 4, "hull_sha256": digest(hull())}]}
        write(out / 'before.json', before); write(out / 'after.json', after)
        diff = fc.verify_stage_difference(before, after, disabled, added)
        diff.update(before_snapshot=record(out / 'before.json'), after_snapshot=record(out / 'after.json'))
        cook = {"algorithm": 'CoACD', "recipe": fc.RECIPE, "original_ultra_vhacd": fc.ORIGINAL_VHACD,
                "equivalent_to_physx_vhacd": False, "cooker_and_usd_imports_isolated": True}
        report = {"schema": fc.FOOT_SCHEMA, "profile": fc.FOOT_PROFILE, "overlay": record(target),
                  "base_robot_asset": {**record(base), "hand_asset_manifest": record(original_hand)},
                  "copied_base_hand_manifest": record(out / 'hand-asset.json'), "original_urdf": record(urdf),
                  "source_meshes": meshes, "cook": cook, "cook_sha256": digest(cook), "sides": sides, "stage_diff": diff}
        write(out / 'foot-asset.json', report)
        binding = {**record(target), 'hand_asset_manifest': record(out / 'hand-asset.json'), 'foot_asset_manifest': record(out / 'foot-asset.json')}
        return target, report, binding, expected, fc.sha(urdf)

    def test_overlay_schema_revalidates_sources_old_hand_receipt_and_diff_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, report, binding, expected, urdf_sha = self.fixture(Path(tmp))
            with patch.dict(fc.FOOT_SHA, expected), patch.object(fc, 'ULTRA_URDF_SHA256', urdf_sha):
                self.assertEqual(fc.validate_overlay(target)['profile'], fc.FOOT_PROFILE)
                self.assertEqual(fc.validate_foot_asset(binding)['overlay']['sha256'], fc.sha(target))
                self.assertEqual((target.parent/'hand-asset.json').read_bytes(), Path(report['base_robot_asset']['hand_asset_manifest']['path']).read_bytes())
                for mutation in ('schema', 'vhacd', 'cook_sha', 'mount', 'hull_sha', 'wrong_side'):
                    bad = deepcopy(report)
                    if mutation == 'schema': bad['schema'] = 'old'
                    elif mutation == 'vhacd': bad['cook']['algorithm'] = 'VHACD'
                    elif mutation == 'cook_sha': bad['cook_sha256'] = '0' * 64
                    elif mutation == 'mount': bad['source_meshes']['left']['collision_origin'] = [0,0,.01]
                    elif mutation == 'hull_sha': bad['sides']['left']['parts'][0]['hull_sha256'] = '0' * 64
                    else: bad['sides']['right']['parts'][0]['path'] = bad['sides']['left']['parts'][0]['path']
                    write(target.parent/'foot-asset.json', bad)
                    with self.assertRaises(ValueError, msg=mutation):
                        fc.validate_overlay(target)

    def test_source_or_copied_hand_receipt_byte_mutation_is_rejected(self):
        for key in ('overlay', 'source_meshes', 'copied_base_hand_manifest', 'base_robot_asset'):
            with tempfile.TemporaryDirectory() as tmp:
                target, report, _, expected, urdf_sha = self.fixture(Path(tmp))
                entry = report[key]['left'] if key == 'source_meshes' else report[key]
                Path(entry['path']).write_bytes(b'changed bytes')
                with patch.dict(fc.FOOT_SHA, expected), patch.object(fc, 'ULTRA_URDF_SHA256', urdf_sha):
                    with self.assertRaises(ValueError, msg=key): fc.validate_overlay(target)

    def test_usd_snapshot_preserves_actual_pxr_types_when_available(self):
        try:
            from pxr import Usd, UsdGeom, UsdPhysics
        except ImportError:
            self.skipTest('Optional pxr unavailable; USD authoring remains a separate runtime validation')
        stage = Usd.Stage.CreateInMemory()
        sphere = UsdGeom.Sphere.Define(stage, '/G1/left_ankle_roll_link/Sphere')
        sphere.CreateRadiusAttr(.005)
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim()).CreateCollisionEnabledAttr(True)
        snapshot = fc._usd_snapshot(stage)
        self.assertEqual(snapshot[str(sphere.GetPath())]['type'], 'Sphere')


if __name__ == '__main__':
    unittest.main()
