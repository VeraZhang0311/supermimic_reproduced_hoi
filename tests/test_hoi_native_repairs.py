"""Synthetic repair provenance tests; no real candidate, approval, USD or GPU job."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import trimesh
from hoi.data import native_repairs as r
from hoi.data.native_assets import BUILD_SCHEMA, bind_compiled_asset, finalize_native_registry, uniform_density_profile
from hoi.v1.native_fixture import read_curated_cooking_fixture
from tests import test_hoi_native_fixture as fixture_support


class NativeRepairTests(unittest.TestCase):
    def setUp(self):
        original=fixture_support.CookingFixtureTests();original.setUp();self.addCleanup(original.doCleanups)
        self.original=original;self.root=original.path.parent
        geometry=self.root/"original_geometry.json";r._write(geometry,original.clip["object_geometry"])
        labels=self.root/"labels";labels.mkdir();(labels/"metadata.json").write_bytes(b"{}");(labels/"labels.npz").write_bytes(b"synthetic labels")
        foot=self.root/"foot.json";foot.write_bytes(b"synthetic foot identity")
        original.fixture["native_robot_asset"]["foot_asset_manifest"]=r._identity(foot)
        robot_recipe=Path(original.fixture["robot_recipe"]["path"])
        robot_recipe.write_bytes(r.canonical({k:original.fixture[k] for k in ("native_robot_asset","reference_kinematics","robot_geometry_sha256")}))
        original.fixture["robot_recipe"]=r._identity(robot_recipe)
        original.fixture["source_record"].update(object_geometry_path=str(geometry),dataset="hoi_retarget",output=str(labels),
            metadata_sha256=r.sha(labels/"metadata.json"),labels_sha256=r.sha(labels/"labels.npz"))
        if self._testMethodName!="test_explicit_source_inertia_is_preserved_without_recomputation":
            build_path=Path(original.fixture["asset"]["build_receipt"]["path"]);old_build=r._json(build_path)
            old_build["physical_profile"]={k:old_build["physical_profile"][k] for k in ("mass","friction","ground_friction")}
            old_build["inertia_mode"]="native_cooking_only_uniform_density";build_path.write_bytes(r.canonical(old_build))
            original.fixture["asset"].update(physical_profile=old_build["physical_profile"],inertia_mode=old_build["inertia_mode"],build_receipt=r._identity(build_path))
        original.write();self.original_fixture_bytes=original.path.read_bytes()
        self.original_build=r._json(original.fixture["asset"]["build_receipt"]["path"])
        label=self.original_build["object_geometry_sha256"];self.group="a"*64;self.other="b"*64
        members=[label]+[r.digest(str(i)) for i in range(2446)];last=r.digest("last_alias")
        base={"build_receipt":original.fixture["asset"]["build_receipt"],"cooking_fixture":r._identity(original.path)}
        registry={"schema":"hoi_curated_native_source_registry_v1","full_selection":True,"selected_geometry_identities":2448,
            "split":{"immutable":"synthetic_original_split"},"object_cooking_groups":[
                {**deepcopy(base),"object_build_group_sha256":self.group,"representative_geometry_sha256":label,"members":members},
                {**deepcopy(base),"object_build_group_sha256":self.other,"representative_geometry_sha256":last,"members":[last]}],
            "bindings":[{"dataset":"hoi_retarget","object_geometry_sha256":m,"object_build_group_sha256":self.group} for m in members]+
                       [{"dataset":"hoi_retarget","object_geometry_sha256":last,"object_build_group_sha256":self.other}]}
        self.registry_path=self.root/"registry.json";r._write(self.registry_path,r._sealed(registry))
        mesh=trimesh.creation.box(extents=[.11,.2,.3]);self.candidate={"shapes":[{"kind":"mesh","vertices":mesh.vertices.tolist(),"faces":mesh.faces.tolist()}]}
        self.candidate_path=self.root/"candidate.json";r._write(self.candidate_path,self.candidate)
        proof=self.root/"proof.json";r._write(proof,{"scope":"synthetic_review_only"})
        self.proposal_path=self.root/"proposal.json"
        self.proposal=r.write_proposal(self.registry_path,self.group,self.candidate_path,proof,
            {k:proof for k in ("builder_hulls","builder_diagnostic","failed_attempt","failed_source")},
            {"operation":"thicken_exposed_planar_component_centered_prism","parameters":{"synthetic_fixture":True}},self.proposal_path)
        self.approval_path=self.root/"synthetic_approval.json"

    def approve(self,**extra):
        approval={"schema":r.APPROVAL_SCHEMA,"approved":True,"authority":"explicit_user_approval",
            "proposal_content_sha256":self.proposal["content_sha256"],"candidate_sha256":self.proposal["candidate_geometry"]["sha256"],
            "scopes":list(r.required_scopes(self.proposal)),"authorization_record":"SYNTHETIC UNIT TEST ONLY; no human approval or executable asset",**extra}
        r._write(self.approval_path,r._sealed(approval))

    @staticmethod
    def fake_usd_writer(geometry,output,*,physical_profile,physical_evidence,collision_approximation,inertia_mode="explicit",preserve_hull_vertices=False):
        # Exercise the real repair/finalize glue without fabricating native
        # compilation: this file is clearly not a loadable USD.
        output=Path(output);output.mkdir(parents=True,exist_ok=False);asset=output/"object.usda";asset.write_bytes(b"SYNTHETIC_NON_NATIVE_TEST_BYTES")
        manifest={"schema":BUILD_SCHEMA,"object_geometry_sha256":r.digest(geometry),"exact_surface_sha256":r.digest(r.exact_surface(geometry["shapes"])),
            "native_asset":str(asset.resolve()),"native_asset_sha256":r.sha(asset),"physical_profile":physical_profile,"physical_evidence":physical_evidence,
            "collision_approximation":collision_approximation,"status":"native_cooking_and_geometry_validation_required","training_ready":False}
        if inertia_mode!="explicit":manifest["inertia_mode"]=inertia_mode
        if preserve_hull_vertices:manifest["hull_vertex_policy"]=r.COOKING_CONTRACT["max_hull_vertices"]
        r._write(output/"build.json",manifest);return manifest

    def build(self):
        self.approve()
        with patch("hoi.data.native_assets.write_native_source",side_effect=self.fake_usd_writer):
            self.fixture=r.write_approved_sources(self.proposal_path,self.approval_path,self.root/"repair")
        return self.fixture

    def test_unapproved_or_wrong_candidate_cannot_create_native_input(self):
        with self.assertRaisesRegex(ValueError,"pending explicit"):
            r.write_approved_sources(self.proposal_path,None,self.root/"must_not_exist")
        self.assertFalse((self.root/"must_not_exist").exists())
        self.approve(candidate_sha256="0"*64)
        with self.assertRaisesRegex(ValueError,"exact proposal/candidate"):
            r.write_approved_sources(self.proposal_path,self.approval_path,self.root/"wrong")

    def test_existing_task_authority_cannot_approve_exposed_geometry_change(self):
        self.approve(authority="existing_task_authorization_for_collision_union_preserving_dedup",containment_proof_sha256=self.proposal["review_proof"]["sha256"])
        with self.assertRaisesRegex(ValueError,"only to the bound"):
            r.require_approval(self.proposal_path,self.approval_path)

    def test_repair_fixture_has_dual_identities_and_v1_remains_strict(self):
        fixture=self.build();f,b,clip=read_curated_cooking_fixture(self.root/"repair/cooking_fixture.json")
        self.assertEqual(f["asset"]["object_geometry_sha256"],self.original_build["object_geometry_sha256"])
        self.assertNotEqual(b["object_geometry_sha256"],f["asset"]["object_geometry_sha256"])
        self.assertEqual(b["physical_profile"],self.proposal["preserved_mass_material"])
        self.assertEqual(self.original.path.read_bytes(),self.original_fixture_bytes)
        altered=deepcopy(fixture);altered["schema"]="hoi_curated_native_cooking_fixture_v1";altered.pop("content_sha256")
        path=self.root/"downgrade.json";r._write(path,r._sealed(altered))
        with self.assertRaisesRegex(ValueError,"explicit v2"):
            read_curated_cooking_fixture(path)
        self.candidate_path.write_bytes(self.candidate_path.read_bytes()+b" ")
        with self.assertRaisesRegex(ValueError,"SHA256 changed"):
            read_curated_cooking_fixture(self.root/"repair/cooking_fixture.json")

    def test_derived_registry_keeps_2448_labels_other_groups_and_old_bytes(self):
        self.build();old_bytes=self.registry_path.read_bytes();old=r._json(self.registry_path)
        derived=r.derive_registry(self.registry_path,[self.root/"repair/cooking_fixture.json"],self.root/"derived.json")
        self.assertEqual(len(derived["bindings"]),2448)
        self.assertEqual({b["object_geometry_sha256"] for b in old["bindings"]},{b["object_geometry_sha256"] for b in derived["bindings"]})
        self.assertEqual(old["object_cooking_groups"][1],derived["object_cooking_groups"][1]);self.assertEqual(self.registry_path.read_bytes(),old_bytes)
        self.assertNotEqual(derived["object_cooking_groups"][0]["object_build_group_sha256"],self.group)
        for mutate in (lambda x:x["bindings"].pop(),lambda x:x["object_cooking_groups"][1]["members"].clear(),lambda x:x["split"].update(tampered=True)):
            bad=deepcopy(derived);mutate(bad)
            with self.assertRaisesRegex(ValueError,"changed original"):
                r.validate_derived_registry(bad)
        with self.assertRaisesRegex(ValueError,"every declared"):
            finalize_native_registry(self.root/"derived.json",{},self.root/"partial_final")

    def test_actual_export_binding_and_component_moments_are_recomputed(self):
        fixture=self.build();build=r._json(fixture["asset"]["build_receipt"]["path"])
        compiled_usd=self.root/"actual_export.usda";compiled_usd.write_bytes(b"SYNTHETIC_NOT_GPU")
        meshes=[{k:v for k,v in m.items() if k!="kind"} for m in self.candidate["shapes"]]
        profile,_=uniform_density_profile(self.candidate,**self.proposal["preserved_mass_material"])
        entry={"source_native_asset":build["native_asset"],"source_native_asset_sha256":build["native_asset_sha256"],
            "native_asset":str(compiled_usd),"native_asset_sha256":r.sha(compiled_usd),"physical_profile":build["physical_profile"],
            "native_convex_evidence":[{"scope":"synthetic test only"}],"collision_meshes":meshes,"repair_binding":fixture["repair_binding"],
            "cooking_input_geometry_sha256":build["object_geometry_sha256"],"cooking_input_surface_sha256":build["exact_surface_sha256"],
            "source_exact_surface_sha256":self.original_build["exact_surface_sha256"]}
        compiled=self.root/"compiled.json";r._write(compiled,{"schema":"hoi_v1_compiled_geometry_recipe_v1","assets":[entry]})
        with self.assertRaisesRegex(ValueError,"requires its approved"):
            bind_compiled_asset(fixture["asset"]["build_receipt"]["path"],compiled,final_output=self.root/"bad")
        with patch("hoi.data.native_assets.write_native_source",side_effect=self.fake_usd_writer):
            final=bind_compiled_asset(fixture["asset"]["build_receipt"]["path"],compiled,final_output=self.root/"final",repair_binding=fixture["repair_binding"])
        self.assertEqual(final["source_build_receipt"],self.original.fixture["asset"]["build_receipt"])
        self.assertEqual(final["cooking_input_build_receipt"],fixture["asset"]["build_receipt"])
        self.assertEqual(final["physical_profile"],profile);r.validate_final_entry(final)
        changed=deepcopy(final);changed["physical_profile"]["com"][0]+=.01
        with self.assertRaisesRegex(ValueError,"COM/inertia"):
            r.validate_final_entry(changed)
        self.approval_path.write_bytes(self.approval_path.read_bytes()+b" ")
        with self.assertRaisesRegex(ValueError,"SHA256 changed"):
            r.validate_final_entry(final)

    def test_full_convex_solid_comparison_allows_interior_points_not_missing_components(self):
        self.build();path=self.root/"repair/repair_binding.json"
        mesh=deepcopy(self.candidate["shapes"][0]);mesh["vertices"].append([0.,0.,0.])
        result=r.compare_candidate_export(path,[mesh]);self.assertTrue(result["passed"])
        with self.assertRaisesRegex(ValueError,"component count"):
            r.compare_candidate_export(path,[])
        mesh["vertices"][0][0]-=.01
        with self.assertRaisesRegex(ValueError,"beyond bound"):
            r.compare_candidate_export(path,[mesh])

    def test_repaired_partial_cooking_keeps_original_label_and_requires_full_registry(self):
        import sys
        from hoi.data import native_cooking as cook
        import runpy
        direct=runpy.run_path(cook.__file__,run_name="native_cooking_direct_script_test")
        self.assertEqual(direct["__package__"],"")
        self.build();source=self.root/"source";(source/"hoi/v1").mkdir(parents=True);(source/"hoi/v1/native.py").write_bytes(b"# synthetic source")
        versions={k:"synthetic" for k in ("torch","warp","mujoco","mujoco_warp","newton")}
        with patch.object(cook,"_runtime_versions",return_value=versions):
            plan=cook.plan_fixtures([self.root/"repair/cooking_fixture.json"],source,sys.executable,["--hand-collision","--self-collision"],
                self.root/"plan",surface_samples=64,defer_surface_report=True)
            direct["plan_fixtures"].__wrapped__.__globals__["_runtime_versions"]=lambda _:versions
            direct_plan=direct["plan_fixtures"]([self.root/"repair/cooking_fixture.json"],source,sys.executable,["--hand-collision","--self-collision"],
                self.root/"direct_plan",surface_samples=64,defer_surface_report=True)
            self.assertEqual(plan["tasks"],direct_plan["tasks"])
            task=plan["tasks"][0]
            self.assertEqual(task["covered_geometry_identities"],[self.original_build["object_geometry_sha256"]])
            binding=r._json(self.root/"repair/repair_binding.json")
            self.assertEqual(task["object_build_group_sha256"],binding["object_build_group_sha256"])
            build=r._json(self.fixture["asset"]["build_receipt"]["path"])
            out=self.root/"synthetic_native";out.mkdir();usd=out/"object.usda";usd.write_bytes(b"SYNTHETIC, not a native model")
            meshes=[{k:v for k,v in m.items() if k!="kind"} for m in self.candidate["shapes"]]
            asset={"source_native_asset_sha256":build["native_asset_sha256"],"native_asset":str(usd),"native_asset_sha256":r.sha(usd),
                "native_convex_evidence":{"synthetic_only":True},"collision_meshes":meshes,"object_geometry_sha256":self.original_build["object_geometry_sha256"],
                "source_exact_surface_sha256":self.original_build["exact_surface_sha256"],"native_compiled_surface_sha256":r.digest(meshes),
                "physical_profile":build["physical_profile"],"repair_binding":self.fixture["repair_binding"],
                "cooking_input_geometry_sha256":build["object_geometry_sha256"],"cooking_input_surface_sha256":build["exact_surface_sha256"],
                "repair_import_comparison":r.compare_candidate_export(self.root/"repair/repair_binding.json",meshes)}
            export=out/"assets.json";r._write(export,{"schema":"hoi_v1_compiled_geometry_recipe_v1","assets":[asset]})
            report={"schema":"hoi_native_geometry_cooking_report_v1","physics_steps":0,"formal_training_started":False,"training_ready":False,
                "native_pair_calibration_complete":False,"num_envs":1,"source":{"fixture_manifest_sha256":task["fixture"]["sha256"],
                "scope":"curated_native_cooking_fixture_not_training","repair_binding":self.fixture["repair_binding"]},
                "native_runtime":{"schema":"hoi_native_cuda_execution_v1","backend":"Newton/MuJoCo-Warp","verified_from_live_buffers":True,
                    "buffer_devices":{k:"cuda:0" for k in ("contact_geom","normal_force","qpos","interval_impulse")},"gpu_name":"SYNTHETIC TEST ONLY","versions":versions},
                "compiled_geometry_export":str(export),"compiled_geometry_export_sha256":r.sha(export),
                "source_exact_surface_sha256":asset["source_exact_surface_sha256"],"native_compiled_surface_sha256":asset["native_compiled_surface_sha256"]}
            r._write(out/"cooking.json",report);r._write(out/"source.json",plan["execution"]["source_files"])
            expected=cook._validate_cook(task,out,plan["execution"]["source_files"],versions)
            self.assertEqual(direct["_validate_cook"](task,out,plan["execution"]["source_files"],versions),expected)
            receipt=cook.adopt_existing_output(self.root/"plan/plan.json",0,out)
            self.assertFalse(receipt["native_acceptance_pass"])
            r.derive_registry(self.registry_path,[self.root/"repair/cooking_fixture.json"],self.root/"derived.json")
            with self.assertRaisesRegex(ValueError,"missing 1 cooked groups"):
                cook.collect_registry_exports(self.root/"derived.json",[self.root/"plan/plan.json"],self.root/"must_not_collect")
            self.assertFalse((self.root/"must_not_collect").exists())
            with self.assertRaisesRegex(ValueError,"missing 1 cooked groups"):
                direct["collect_registry_exports"](self.root/"derived.json",[self.root/"plan/plan.json"],self.root/"direct_must_not_collect")
            report["source"]["repair_binding"]["sha256"]="0"*64
            (out/"cooking.json").write_bytes(r.canonical(report))
            with self.assertRaisesRegex(ValueError,"source bindings"):
                cook._validate_cook(task,out,plan["execution"]["source_files"],versions)
            with self.assertRaisesRegex(ValueError,"source bindings"):
                direct["_validate_cook"](task,out,plan["execution"]["source_files"],versions)

    def test_explicit_source_inertia_is_preserved_without_recomputation(self):
        fixture=self.build();build=r._json(fixture["asset"]["build_receipt"]["path"])
        self.assertNotIn("inertia_mode",build)
        self.assertEqual(build["physical_profile"],self.original_build["physical_profile"])
        self.assertEqual(r.required_scopes(self.proposal),("native_collision_geometry","preserve_explicit_source_inertia"))
        read_curated_cooking_fixture(self.root/"repair/cooking_fixture.json")
        usd=self.root/"actual.usda";usd.write_bytes(b"synthetic actual import only")
        entry={"source_native_asset":build["native_asset"],"source_native_asset_sha256":build["native_asset_sha256"],
            "native_asset":str(usd),"native_asset_sha256":r.sha(usd),"physical_profile":build["physical_profile"],
            "native_convex_evidence":[{"synthetic":True}],"collision_meshes":[{k:v for k,v in self.candidate["shapes"][0].items() if k!="kind"}],
            "repair_binding":fixture["repair_binding"],"cooking_input_geometry_sha256":build["object_geometry_sha256"],
            "cooking_input_surface_sha256":build["exact_surface_sha256"],"source_exact_surface_sha256":self.original_build["exact_surface_sha256"]}
        export=self.root/"explicit_export.json";r._write(export,{"schema":"hoi_v1_compiled_geometry_recipe_v1","assets":[entry]})
        with patch("hoi.data.native_assets.write_native_source",side_effect=self.fake_usd_writer),patch("hoi.data.native_assets.uniform_density_profile",side_effect=lambda geometry,**kw: uniform_density_profile(geometry,**kw) if kw["mass"]==1. else (_ for _ in ()).throw(AssertionError("explicit source moments must not be rederived"))):
            final=bind_compiled_asset(fixture["asset"]["build_receipt"]["path"],export,final_output=self.root/"final",repair_binding=fixture["repair_binding"])
        self.assertEqual(final["physical_profile"],self.original_build["physical_profile"])
        r.validate_final_entry(final)

    def test_collision_only_containment_is_recomputed_and_visuals_cannot_authorize_removal(self):
        import numpy as np
        mesh=self.candidate["shapes"][0];plane=[[0.,0.,0.],[.001,0.,0.],[0.,.001,0.]]
        archive=self.root/"builder.npz"
        np.savez(archive,vertices_0000=np.asarray(mesh["vertices"])*100,faces_0000=mesh["faces"],
            vertices_0001=mesh["vertices"],faces_0001=mesh["faces"],vertices_0002=plane,faces_0002=[[0,1,2]])
        diagnostic=self.root/"builder.json";r._write(diagnostic,{"all_hull_stats":[{"shape":0,"collision_enabled":False},{"shape":1,"collision_enabled":True},{"shape":2,"collision_enabled":True}]})
        identities={"builder_hulls":r._identity(archive),"builder_diagnostic":r._identity(diagnostic)}
        recipe={"removed_collision_shapes":[2],"containers":{"2":1}}
        proof=r.verify_redundant_removal(self.candidate,identities,recipe)
        self.assertTrue(proof["collision_union_preserved"]);self.assertFalse(proof["visuals_used"])
        self.assertLess(proof["removed"][0]["maximum_signed_halfspace_m"],-.05)
        with self.assertRaisesRegex(ValueError,"visual/removed"):
            r.verify_redundant_removal(self.candidate,identities,{**recipe,"containers":{"2":0}})
        changed=deepcopy(self.candidate);changed["shapes"][0]["vertices"][0][0]+=.001
        with self.assertRaisesRegex(ValueError,"retained collision"):
            r.verify_redundant_removal(changed,identities,recipe)

    def test_ten_micron_component_collapse_and_expansion_fail_even_inside_absolute_tolerance(self):
        import numpy as np
        mesh=trimesh.creation.box(extents=[.01,.02,1e-5]);candidate={"shapes":[{"kind":"mesh","vertices":mesh.vertices.tolist(),"faces":mesh.faces.tolist()}]}
        fake_binding=(None,{"candidate_geometry":{"sha256":"synthetic"}},None,None,None,None,candidate)
        # .9/1.1 change the surface by only .5 microns, below the 1 micron
        # plane tolerance; separate physical volume/width bounds must reject.
        with patch.object(r,"read_binding",return_value=fake_binding):
            report=r.compare_candidate_export("synthetic",candidate["shapes"])
            self.assertEqual(report["maximum_component_volume_relative_error"],0.)
            for factor in (.1,.9,1.1,1.9):
                changed=deepcopy(candidate["shapes"]);v=np.asarray(changed[0]["vertices"]);v[:,2]*=factor;changed[0]["vertices"]=v.tolist()
                with self.subTest(factor=factor),self.assertRaisesRegex(ValueError,"beyond.*bound"):
                    r.compare_candidate_export("synthetic",changed)
