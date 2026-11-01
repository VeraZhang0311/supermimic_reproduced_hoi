"""Recorded-transform, geometry-source and full-frame renderer contracts."""
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hoi.v1.evaluate import ROLLOUT_SCHEMA
from hoi.v1.render import (load_saved, fixed_camera_parameters, _triangles_ascii_usda, write_stl,
                           object_display, apply_saved_frame, verify_decoded_frames, sha, robot_visual_assets)


def saved_fixture(path):
    n = 3
    body = torch.zeros(n, 39, 3)
    body[:, :, 0] = torch.arange(n)[:, None] + 2  # visible drift must remain visible
    rotations = torch.zeros(n, 39, 4)
    rotations[..., 3] = 1
    obj = torch.zeros(n, 13)
    obj[:, 3] = 1
    reference = torch.zeros(n, 630)
    reference[:, 77] = 1
    reference[:, 201:357] = rotations.flatten(-2)
    data = {"schema_version": ROLLOUT_SCHEMA,
            "states": {"body_pos": body, "body_rot": rotations, "o": obj, "reference_source630": reference},
            "control": {"action": torch.zeros(n - 1, 29)}, "time": torch.arange(n, dtype=torch.float64) / 50,
            "metadata": {"body_names": [f"b{i}" for i in range(39)], "asset_id": 0, "name": "synthetic_render_contract"}}
    torch.save(data, path)
    return data


class RenderTest(unittest.TestCase):
    def test_relocated_URDF_requires_explicit_mesh_root_and_keeps_hand_SHA_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            metadata=root/"metadata"
            assets=root/"actual_visual_asset"
            metadata.mkdir()
            (assets/"meshes").mkdir(parents=True)
            names=["left_rubber_hand","right_rubber_hand"]
            files={side:assets/"meshes"/(side+"_rubber_hand.STL") for side in ("left","right")}
            for side,path in files.items():
                path.write_bytes(("synthetic_mesh_bytes_"+side).encode())
            urdf=metadata/"unchanged_virtual.urdf"
            urdf.write_text('<robot name="relocated">'+''.join(
                f'<link name="{name}"><visual><origin xyz="0 0 0" rpy="0 0 0"/>'
                f'<geometry><mesh filename="meshes/{name}.STL" scale="1 1 1"/></geometry></visual></link>' for name in names)+'</robot>')
            original=urdf.read_bytes()
            with patch("hoi.hand_collision.HAND_SHA",{side:sha(path) for side,path in files.items()}):
                with self.assertRaisesRegex(FileNotFoundError,"explicit --robot-mesh-root"):
                    robot_visual_assets(urdf,names)
                specs,visuals,record=robot_visual_assets(urdf,names,mesh_root=assets)
                self.assertTrue(record["relative_visual_mesh_root_explicit"])
                self.assertEqual(record["relative_visual_mesh_root"],str(assets.resolve()))
                self.assertTrue(record["real_hand_STL_verified"])
                self.assertEqual(urdf.read_bytes(),original)
                for spec,source in zip(specs,record["visual_meshes"]):
                    self.assertEqual(Path(spec["file"]),Path(source["path"]))
                    self.assertEqual(source["sha256"],sha(spec["file"]))
                    self.assertEqual(spec["scale"],"1 1 1")
                    self.assertEqual(visuals[source["body"]][0]["pos"],"0 0 0")
                files["right"].write_bytes(b"wrong_hand_mesh")
                with self.assertRaisesRegex(ValueError,"byte-verified STL"):
                    robot_visual_assets(urdf,names,mesh_root=assets)

    def test_saved_Tplus1_unaligned_world_transforms_and_clock(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rollout.pt"
            original = saved_fixture(path)
            result = load_saved(path, sha(path))
            np.testing.assert_equal(result["actual_pos"], original["states"]["body_pos"].numpy())
            np.testing.assert_equal(result["reference_pos"], np.zeros((3, 39, 3)))
            self.assertEqual(len(result["time"]), result["control_steps"] + 1)
            self.assertEqual(result["actual_pos"][0, 0, 0], 2.)
            self.assertEqual(result["actual_pos"][-1, 0, 0], 4.)
            camera = fixed_camera_parameters(result)
            self.assertEqual(camera["lookat"][0], 2.)
            self.assertTrue(camera["shared_world"])
            self.assertEqual(camera["trajectory_alignment"], "none")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_saved(path, "0" * 64)
            original["time"][-1] = .05
            torch.save(original, path)
            with self.assertRaisesRegex(ValueError, "clock"):
                load_saved(path)

    def test_incomplete_or_invalid_saved_transforms_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rollout.pt"
            data = saved_fixture(path)
            data["states"]["body_pos"] = data["states"]["body_pos"][:-1]
            torch.save(data, path)
            with self.assertRaisesRegex(ValueError, "T\+1"):
                load_saved(path)
            data = saved_fixture(path)
            data["states"]["body_rot"][0, 0] = 0
            torch.save(data, path)
            with self.assertRaisesRegex(ValueError, "quaternion"):
                load_saved(path)

    def test_curated_saved_reference_transforms_preserve_all_frames_and_quaternions(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "curated.pt"
            data = saved_fixture(path)
            states = data["states"]
            states.pop("reference_source630")
            states["reference_body_pos"] = torch.arange(3 * 39 * 3, dtype=torch.float32).reshape(3, 39, 3) / 100
            quat_xyzw = torch.tensor([1., 2., 3., 4.]) / torch.sqrt(torch.tensor(30.))
            states["reference_body_rot"] = quat_xyzw.expand(3, 39, 4).clone()
            states["reference_object"] = torch.zeros(3, 13)
            states["reference_object"][:, :3] = torch.tensor([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]])
            states["reference_object"][:, 3:7] = quat_xyzw[[3, 0, 1, 2]]
            # Deliberately inconsistent joint/root data: display uses saved
            # transforms exclusively and does not infer reference poses.
            states["reference_robot"] = torch.full((3, 71), 100.)
            torch.save(data, path)
            with patch("hoi.ultra_kinematics.UltraKinematics.forward", side_effect=AssertionError("renderer must not invoke FK")), \
                    patch("hoi.v1.networks.Policy.forward", side_effect=AssertionError("renderer must not invoke policy")):
                result = load_saved(path)
            np.testing.assert_array_equal(result["reference_pos"], states["reference_body_pos"].numpy())
            np.testing.assert_array_equal(result["reference_quat"], states["reference_body_rot"][..., [3, 0, 1, 2]].numpy())
            np.testing.assert_array_equal(result["reference_object_pos"], states["reference_object"][:, :3].numpy())
            np.testing.assert_array_equal(result["reference_object_quat"], states["reference_object"][:, 3:7].numpy())
            np.testing.assert_array_equal(result["actual_pos"], states["body_pos"].numpy())
            self.assertEqual(result["reference_display_source"], "saved_reference_body_and_object_transforms")
            self.assertEqual(len(result["time"]), 3)
            self.assertEqual(result["control_steps"], 2)
            self.assertNotIn("reference_source630", states)

    def test_curated_missing_truncated_or_invalid_reference_never_fabricated(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "curated.pt"
            data = saved_fixture(path)
            states = data["states"]
            states.pop("reference_source630")
            states["reference_body_pos"] = states["body_pos"].clone()
            states["reference_body_rot"] = states["body_rot"].clone()
            states["reference_object"] = states["o"].clone()
            for field in ("reference_body_pos", "reference_body_rot", "reference_object"):
                with self.subTest(field=field):
                    original = states.pop(field)
                    torch.save(data, path)
                    with self.assertRaisesRegex(ValueError, "cannot reconstruct"):
                        load_saved(path)
                    states[field] = original[:-1]
                    torch.save(data, path)
                    with self.assertRaisesRegex(ValueError, "T\+1"):
                        load_saved(path)
                    states[field] = original
            states["reference_body_rot"][0, 0] = 0
            torch.save(data, path)
            with self.assertRaisesRegex(ValueError, "quaternion"):
                load_saved(path)
            states["reference_body_rot"] = states["body_rot"].clone()
            states["reference_object"][0, 0] = float("nan")
            torch.save(data, path)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                load_saved(path)

    def test_saved_supports_use_recorded_actual_and_reset_held_reference(self):
        from hoi.v1.scene_supports import SCENE_SUPPORT_SCHEMA,SUPPORT_IDS,omni_support_recipe,digest
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"supports.pt";payload=saved_fixture(path)
            pose=torch.zeros(3,2,7);pose[...,6]=1
            pose[:,:,2]=torch.tensor([.8,-5.])
            actual=pose.clone();actual[1:,0,0]=1e-6
            payload["states"].update(support_pose_xyzw=actual,reference_support_pose_xyzw=pose,support_present=torch.ones(3,2,dtype=torch.bool))
            payload["metadata"]["scene_support_display"]={"schema":SCENE_SUPPORT_SCHEMA,"support_ids":list(SUPPORT_IDS),
                "geometry_recipe":omni_support_recipe(),"geometry_recipe_sha256":digest(omni_support_recipe()),"quaternion":"xyzw"}
            torch.save(payload,path);data=load_saved(path)
            np.testing.assert_array_equal(data["actual_support_pos"],actual[...,:3].numpy())
            np.testing.assert_array_equal(data["reference_support_pos"],pose[...,:3].numpy())
            self.assertEqual(data["support_ids"],list(SUPPORT_IDS))
            self.assertGreater(fixed_camera_parameters(data)["lookat"][2],-1.)
            payload["states"]["reference_support_pose_xyzw"][1,0,0]=.1
            torch.save(payload,path)
            with self.assertRaisesRegex(ValueError,"held"):load_saved(path)
            del payload["states"]["reference_support_pose_xyzw"]
            torch.save(payload,path)
            with self.assertRaisesRegex(ValueError,"cannot be reconstructed"):load_saved(path)

    def test_support_display_model_has_no_dynamics_and_uses_saved_pose(self):
        import mujoco
        from hoi.v1.render import display_model
        names=[f"b{i}" for i in range(39)]
        with patch("hoi.v1.render.robot_visual_assets",return_value=([],{},{})):
            model,maps,_=display_model("unused",[],names,support_ids=("table1",),support_size_m=[.3,.25,.01])
        self.assertEqual((model.nq,model.nv),(0,0))
        sim=mujoco.MjData(model)
        data={}
        for prefix in ("actual","reference"):
            data[prefix+"_pos"]=np.zeros((1,39,3));data[prefix+"_quat"]=np.tile([1,0,0,0],(1,39,1))
            data[prefix+"_object_pos"]=np.zeros((1,3));data[prefix+"_object_quat"]=np.array([[1,0,0,0]])
            data[prefix+"_support_pos"]=np.array([[[.2,.3,.8]]]);data[prefix+"_support_quat"]=np.array([[[1,0,0,0]]])
        with patch("mujoco.mj_step",side_effect=AssertionError("offline renderer cannot step physics")):
            apply_saved_frame(model,sim,maps,data,0)
        np.testing.assert_allclose(sim.mocap_pos[maps["actual_support"]],[[.2,.3,.8]])

    def test_multi_hull_USD_metadata_and_transform_guard(self):
        mesh = '''def Mesh "NAME" (apiSchemas = ["PhysicsCollisionAPI"]) {
point3f[] points = [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
int[] faceVertexCounts = [3,3,3,3]
int[] faceVertexIndices = [0,2,1,0,1,3,0,3,2,1,2,3]
}'''
        prefix = '#usda 1.0\n(metersPerUnit=1 upAxis="Z")\ndef Xform "Object" {\n'
        text = prefix + mesh.replace("NAME", "a") + mesh.replace("NAME", "b") + "\n}"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "hulls.usda"
            path.write_text(text)
            triangles, record = _triangles_ascii_usda(path)
            self.assertEqual(triangles.shape, (8, 3, 3))
            self.assertEqual(record["mesh_count"], 2)
            stl = Path(temp) / "mesh.stl"
            write_stl(stl, triangles)
            self.assertEqual(struct.unpack_from("<I", stl.read_bytes(), 80)[0], 8)
            self.assertEqual(stl.stat().st_size, 84 + 50 * 8)
            for bad in ('double3 xformOp:translate = (1,0,0)', 'float xformOp:rotateX = 1'):
                path.write_text(text[:-1] + bad + "\n}")
                with self.assertRaisesRegex(ValueError, "transforms"):
                    _triangles_ascii_usda(path)

    def test_collision_display_requires_explicit_request_and_bound_bytes(self):
        mesh = {"vertices": torch.tensor([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]]),
                "faces": torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "geometry.pt"
            torch.save({"collision_meshes": [mesh, mesh]}, cache)
            asset = {"geometry_cache": {"path": str(cache), "sha256": sha(cache)}, "collision_hash": "synthetic"}
            with self.assertRaisesRegex(ValueError, "explicitly select"):
                object_display(asset, root / "no_silent_fallback")
            paths, record = object_display(asset, root / "explicit", use_collision_geometry=True)
            self.assertEqual(record["role"], "explicit_collision_geometry_display")
            self.assertEqual(record["mesh_count"], 2)
            self.assertEqual(record["triangle_count"], 8)
            self.assertTrue(paths[0].is_file())
            cache.write_bytes(cache.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                object_display(asset, root / "changed", use_collision_geometry=True)

    def test_frame_application_assigns_actual_and_reference_without_physics(self):
        data = {"actual_pos": np.full((1, 39, 3), 2.), "reference_pos": np.zeros((1, 39, 3)),
                "actual_quat": np.tile([1., 0, 0, 0], (1, 39, 1)), "reference_quat": np.tile([1., 0, 0, 0], (1, 39, 1)),
                "actual_object_pos": np.array([[3., 0, 1]]), "reference_object_pos": np.array([[0., 0, 1]]),
                "actual_object_quat": np.array([[1., 0, 0, 0]]), "reference_object_quat": np.array([[1., 0, 0, 0]])}
        maps = {"actual": np.arange(39), "reference": np.arange(39, 78), "actual_object": 78, "reference_object": 79}
        state = SimpleNamespace(mocap_pos=np.zeros((80, 3)), mocap_quat=np.zeros((80, 4)))
        calls = []
        fake_mujoco = SimpleNamespace(mj_kinematics=lambda model, state: calls.append("kinematics"),
                                     mj_step=lambda *args: self.fail("physics stepping is forbidden"))
        with patch.dict("sys.modules", {"mujoco": fake_mujoco}):
            apply_saved_frame(None, state, maps, data, 0)
        np.testing.assert_equal(state.mocap_pos[:39], 2.)
        np.testing.assert_equal(state.mocap_pos[39:78], 0.)
        np.testing.assert_equal(state.mocap_pos[78], [3., 0, 1])
        self.assertEqual(calls, ["kinematics"])

    def test_decode_verification_rejects_a_missing_last_frame(self):
        with patch("hoi.v1.render.subprocess.run", return_value=SimpleNamespace(stdout="frame=286\nprogress=end\n", stderr="")):
            self.assertEqual(verify_decoded_frames("ffmpeg", "sample.mp4", 286), 286)
        with patch("hoi.v1.render.subprocess.run", return_value=SimpleNamespace(stdout="frame=285\nprogress=end\n", stderr="")):
            with self.assertRaisesRegex(RuntimeError, "frame count"):
                verify_decoded_frames("ffmpeg", "sample.mp4", 286)


if __name__ == "__main__":
    unittest.main()
