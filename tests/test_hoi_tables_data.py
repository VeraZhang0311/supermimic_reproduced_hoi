"""Tables timebase, whitelist, leakage protection and runtime cache checks."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from hoi_tables.data import (ASSETS,CACHE_SCHEMA,SCHEMA,TIMEBASE,TableMotionLib,
                            adapt_row,assign_partitions,content_hash,read_manifest,
                            read_selection,sha256_file,trajectory_hash)
from supermimic.robot_metadata import G1_JOINT_NAMES
from hoi_tables.assets import CATALOG_SCHEMA,load_assets,load_geometry_cache
from hoi_residual.assets import _digest_json
from hoi_residual.shape import point_sha256


def row(name="sub1_largetable_001",n=7,offset=0.):
    time = torch.arange(n,dtype=torch.float64)/30
    p = torch.stack((time+offset,2*time,torch.ones_like(time)),1)
    q = torch.zeros(n,29,dtype=torch.float64)
    q[:,0] = time
    quat = torch.zeros(n,4,dtype=torch.float64);quat[:,0] = 1
    return {"subject":name,"sequence_id":"omomo/"+name,"clip_id":"omomo/unitree_g1/"+name,
            "object":name.split("_")[1],"fps":30.,"n_frames":n,
            "dof_names":list(G1_JOINT_NAMES),"root_pos":p.tolist(),"root_rot":quat.tolist(),
            "dof_pos":q.tolist(),"object_pos":(p+.3).tolist(),"object_rot":quat.tolist(),
            "per_link_contact_flags":[[True,False,False,False]]*n}


def fixture_manifest(root,rows,roles=None):
    roles = roles or {rows[0]["subject"]:"train",rows[1]["subject"]:"protected_eval"}
    cache = {"schema":CACHE_SCHEMA,"timebase":TIMEBASE,"clips":{}}
    clips = []
    for source in rows:
        value = adapt_row(source)
        cache["clips"][source["clip_id"]] = value
        clips.append({"clip_id":source["clip_id"],"subject":source["subject"],"source_group":source["sequence_id"],
                      "object_asset":source["object"],"source_trajectory_sha256":trajectory_hash(source),
                      "state_frames":len(value["states"]),"partition":roles[source["subject"]],"role":roles[source["subject"]]})
    cache_path = root/"reference_50hz.pt";torch.save(cache,cache_path)
    m = {"schema":SCHEMA,"object_assets":list(ASSETS),"timebase":TIMEBASE,"target_joint_names":list(G1_JOINT_NAMES),
         "clips":clips,"selection":{"included_count":len(clips)},"cache":{"path":cache_path.name,"sha256":sha256_file(cache_path)}}
    m["content_sha256"] = content_hash(m)
    path = root/"manifest.json";path.write_text(json.dumps(m),encoding="utf-8")
    return path,m


class TableDataTests(unittest.TestCase):
    def test_exact_user_whitelist(self):
        selection = read_selection(Path(__file__).resolve().parents[1]/"configs/hoi_tables_selection.json")
        self.assertEqual(len(selection["included"]),226)
        for obj in ASSETS:
            self.assertEqual(sum(name.split("_")[1] == obj for name in selection["included"]),113)
        self.assertEqual(set(selection["excluded"]),{"sub17_smalltable_047","sub1_largetable_049","sub6_largetable_036"})

    def test_resamples_30_to_50_without_scaling_trajectories(self):
        source = row(n=7)
        value = adapt_row(source)
        self.assertEqual(value["states"].shape,(11,71))
        self.assertEqual(value["objects"].shape,(11,13))
        time = torch.arange(11)*.02
        torch.testing.assert_close(value["states"][:,0],time)
        torch.testing.assert_close(value["objects"][:,0],time+.3)
        torch.testing.assert_close(value["states"][:,7],torch.ones(11))
        torch.testing.assert_close(value["states"][:,42],torch.ones(11))
        torch.testing.assert_close(value["states"][:,3],torch.ones(11))
        self.assertEqual(value["source_contacts"].shape,(11,4))

    def test_fractional_endpoint_is_not_extrapolated(self):
        value = adapt_row(row(n=5))
        self.assertEqual(len(value["states"]),7)
        self.assertAlmostEqual(float(value["states"][-1,0]),.12,places=6)
        self.assertLessEqual(float(value["states"][-1,0]),4/30)

    def test_named_joint_reindexing(self):
        source = row()
        source["dof_names"] = source["dof_names"][::-1]
        source["dof_pos"] = [frame[::-1] for frame in source["dof_pos"]]
        torch.testing.assert_close(adapt_row(source)["states"],adapt_row(row())["states"])

    def test_wxyz_slerp_and_derived_angular_velocity(self):
        source = row(n=7)
        t = torch.arange(7,dtype=torch.float64)/30
        quat = torch.stack((torch.cos(t),torch.zeros_like(t),torch.zeros_like(t),torch.sin(t)),1)
        source["object_rot"] = quat.tolist()
        actual = adapt_row(source)["objects"]
        torch.testing.assert_close(actual[:,12],torch.full((11,),2.),rtol=1e-4,atol=1e-4)
        torch.testing.assert_close(actual[:,3],torch.cos(torch.arange(11)*.02),rtol=1e-5,atol=1e-5)

    def test_rejects_invalid_timebase_joint_names_and_quaternion(self):
        for mutate in (lambda r:r.update(fps=60.),lambda r:r["dof_names"].__setitem__(0,"bad_joint"),
                       lambda r:r["root_rot"][0].__setitem__(0,0.),lambda r:r["root_pos"][0].__setitem__(0,float("nan"))):
            source = row();mutate(source)
            with self.assertRaises(ValueError):adapt_row(source)

    def test_deterministic_stratified_group_split(self):
        rows = [row(f"sub{s}_{obj}_{i:03d}",offset=s*100+i) for obj in ASSETS for s in (1,2,3) for i in range(10)]
        actual = assign_partitions(rows)
        self.assertEqual(actual,assign_partitions(list(reversed(rows))))
        for obj in ASSETS:
            self.assertEqual(sum(actual[r["subject"]] == "protected_eval" for r in rows if r["object"] == obj),3)
        duplicate = deepcopy(rows[0]);duplicate["subject"]="sub9_largetable_999";duplicate["sequence_id"]="omomo/"+duplicate["subject"]
        with_duplicate = assign_partitions(rows+[duplicate])
        self.assertEqual(with_duplicate[duplicate["subject"]],with_duplicate[rows[0]["subject"]])

    def test_runtime_api_bins_future_endpoint_and_eval_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_ = fixture_manifest(Path(tmp),[row(n=32),row("sub2_smalltable_001",n=8,offset=1.)])
            motion = TableMotionLib(path,None,"cpu")
            self.assertTrue(motion.allowed_for_training)
            self.assertEqual(motion.dataset,"hoi_retarget_tables")
            self.assertEqual(motion.num_bins,2)
            self.assertTrue(torch.all(motion.bin_start < motion.clip_end[motion.bin_clip]))
            end = motion.clip_end[:1]
            self.assertTrue(torch.equal(motion.future_frames(end,10),end[:,None].expand(-1,10)))
            self.assertEqual(motion.goal(torch.zeros(1,dtype=torch.long),10).shape,(1,730))
            self.assertEqual(motion.object_ref(torch.zeros(1,dtype=torch.long))[0].shape,(1,3))
            self.assertEqual(motion.compatible_bins("largetable").numel(),2)
            with self.assertRaisesRegex(ValueError,"protected eval"):
                TableMotionLib(path,None,"cpu",partition="protected_eval")
            eval_motion = TableMotionLib(path,None,"cpu",partition="protected_eval",for_training=False)
            self.assertFalse(eval_motion.allowed_for_training)
            self.assertEqual(eval_motion.names,["sub2_smalltable_001"])

    def test_cache_tamper_and_duplicate_capture_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,m = fixture_manifest(Path(tmp),[row(),row("sub2_smalltable_001",offset=1.)])
            m["clips"][1]["source_trajectory_sha256"] = m["clips"][0]["source_trajectory_sha256"]
            m["content_sha256"] = content_hash(m);path.write_text(json.dumps(m),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"leakage"):read_manifest(path)
            path,m = fixture_manifest(Path(tmp),[row(),row("sub2_smalltable_001",offset=1.)])
            with (Path(tmp)/"reference_50hz.pt").open("ab") as stream:stream.write(b"changed")
            with self.assertRaisesRegex(ValueError,"cache SHA256"):
                TableMotionLib(path,None,"cpu")

    def test_geometry_cache_is_bound_to_current_table_colliders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh = {"vertices":[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
                    "faces":[[0,2,1],[0,1,3],[0,3,2],[1,2,3]],"frame":"object_local"}
            points = torch.arange(384,dtype=torch.float32).reshape(128,3)/1000
            cache = {"points":points,"sdf_grid":torch.zeros(2,2,2),"grid_min":torch.zeros(3),
                     "grid_max":torch.ones(3),"bbox_diagonal":torch.tensor(3.).sqrt(),
                     "collision_meshes":[mesh],"metadata":{"frame":"object_link_local_metres","point_count":128,
                     "collision_hash":"bound","exact_scale":[1.,1.,1.],"grid_resolution":2,"grid_axis_order":"zyx",
                     "sdf_sign":"negative_inside_collider_union","point_hash":point_sha256(points)}}
            path = root/"geometry.pt";torch.save(cache,path)
            native = root/"native.usda";native.write_text("fixture bytes; never imported",encoding="utf-8")
            asset = {"asset_id":0,"object_asset":"largetable","native_asset":str(native),"native_asset_sha256":sha256_file(native),
                     "collision_meshes":[mesh],"collision_hash":"bound","exact_scale":[1.,1.,1.],
                     "geometry_cache":{"path":str(path),"sha256":sha256_file(path)},
                     "physical_profile":{"mass":6.,"com":[0.,0.,0.],"inertia":[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],"friction":.9,"ground_friction":.3}}
            torch.testing.assert_close(load_geometry_cache(asset)["points"],points)
            changed = deepcopy(asset);changed["collision_meshes"][0]["vertices"][0][0] = .1
            with self.assertRaisesRegex(ValueError,"vertices differ"):
                load_geometry_cache(changed)
            catalog = {"schema_version":CATALOG_SCHEMA,"assets":[asset,dict(asset,asset_id=1,object_asset="smalltable")]}
            catalog["catalog_content_sha256"] = _digest_json(catalog)
            catalog_path = root/"assets.json";catalog_path.write_text(json.dumps(catalog),encoding="utf-8")
            self.assertEqual([a["object_asset"] for a in load_assets(catalog_path)],["largetable","smalltable"])
            native.write_text("changed native collider bytes",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"native_asset bytes changed"):
                load_assets(catalog_path)


if __name__ == "__main__":
    unittest.main()
