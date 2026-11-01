"""Explicit approved collision-repair lineage; original labels/builds stay immutable.

Proposal creation is review-only. Writing a native input, deriving a registry,
loading a repair fixture, and finalizing an asset all require the separately
recorded authorization of the exact proposal and candidate file bytes. Existing
task authority is restricted to independently recomputed redundant containment.
"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from hoi.data.native_assets import canonical, digest, sha, exact_surface

PROPOSAL_SCHEMA="hoi_native_collision_repair_proposal_v1"
APPROVAL_SCHEMA="hoi_native_collision_repair_approval_v1"
BINDING_SCHEMA="hoi_native_collision_repair_binding_v1"
FIXTURE_SCHEMA="hoi_curated_native_repair_cooking_fixture_v2"
SCOPES=("native_collision_geometry","recompute_uniform_density_inertia")

def required_scopes(proposal):
    return SCOPES if proposal["source_inertia_mode"]=="native_cooking_only_uniform_density" else ("native_collision_geometry","preserve_explicit_source_inertia")
OPERATIONS=("remove_contained_degenerate_component","merge_degenerate_component_with_adjacent_convex_hull",
            "thicken_exposed_planar_component_centered_prism")
COOKING_CONTRACT={"approximation":"convexHull","max_hull_vertices":"per_component_input_vertex_count_no_decimation",
                  "missing_inertia":"uniform_density_actual_compiled_closed_components",
                  "overlap":"component_volume_sum_not_boolean_union", "native_reimport_tolerance_m":1e-6, "component_volume_relative_tolerance":1e-3,
                  "component_minimum_facet_width_relative_tolerance":1e-3}


def _json(path):return json.loads(Path(path).read_bytes())
def _identity(path):return {"path":str(Path(path).resolve()),"sha256":sha(path)}
def _verify(binding):
    if not isinstance(binding,dict) or sha(binding["path"])!=binding["sha256"]:raise ValueError("repair dependency SHA256 changed")
    return binding

def _sealed(value):return {**value,"content_sha256":digest(value)}
def _read(path,schema):
    value=_json(path)
    if value.get("schema")!=schema or value.get("content_sha256")!=digest({k:v for k,v in value.items() if k!="content_sha256"}):
        raise ValueError("repair schema/content identity mismatch")
    return value

def _write(path,value):
    with Path(path).open("xb") as stream:stream.write(canonical(value)+b"\n")


def _candidate(path):
    from hoi.data.native_assets import uniform_density_profile
    geometry=_json(path)
    if set(geometry)!={"shapes"} or not geometry["shapes"] or any(set(s)!={"kind","vertices","faces"} or s["kind"]!="mesh" for s in geometry["shapes"]):
        raise ValueError("repair candidate requires explicit object-local metric mesh components only")
    # CPU validity is not a native cooking/equivalence claim. Actual final
    # density moments are derived again from the true imported/exported hulls.
    uniform_density_profile(geometry,mass=1.,friction=0.,ground_friction=0.)
    import numpy as np
    import trimesh
    from scipy.spatial import ConvexHull
    # USD convexHull declares the convex solid of these exact input vertices.
    # Trimesh's adjacency-angle heuristic rejects near-coplanar float32 cooked
    # facets. Instead audit every original face against the declared solid;
    # neither input vertices nor triangulation are rewritten by this check.
    for shape in geometry["shapes"]:
        vertices=np.asarray(shape["vertices"],dtype=float);faces=np.asarray(shape["faces"],dtype=int)
        mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False);hull=ConvexHull(vertices)
        triangles=vertices[faces];normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);length=np.linalg.norm(normals,axis=-1)
        if not np.isfinite(length).all() or bool((length<=0).any()):raise ValueError("repair candidate contains a degenerate triangle")
        normals/=length[:,None]
        normals*=np.where(np.einsum("ij,ij->i",normals,triangles.mean(1)-vertices.mean(0))<0,-1,1)[:,None]
        planes=np.einsum("ij,ij->i",normals,triangles[:,0])
        violation=float((vertices@normals.T-planes).max())
        volume_error=abs(abs(float(mesh.volume))/float(hull.volume)-1.)
        if violation>COOKING_CONTRACT["native_reimport_tolerance_m"] or volume_error>COOKING_CONTRACT["component_volume_relative_tolerance"]:
            raise ValueError("repair input triangulation differs materially from its declared convex solid")
    return geometry


def _registry(path):
    registry=_read(path,"hoi_curated_native_source_registry_v1")
    labels=[x["object_geometry_sha256"] for x in registry["bindings"]]
    if not registry.get("full_selection") or len(labels)!=registry["selected_geometry_identities"] or len(set(labels))!=len(labels):
        raise ValueError("repair derivation requires complete unique original registry coverage")
    groups={g["object_build_group_sha256"]:g for g in registry["object_cooking_groups"]}
    if len(groups)!=len(registry["object_cooking_groups"]):raise ValueError("duplicate source native groups")
    for key,group in groups.items():
        members=[b["object_geometry_sha256"] for b in registry["bindings"] if b.get("object_build_group_sha256")==key]
        if len(set(group["members"]))!=len(group["members"]) or set(members)!=set(group["members"]) or group["representative_geometry_sha256"] not in group["members"]:
            raise ValueError("source registry group membership/coverage mismatch")
    if any(b.get("dataset")!="ultra" and b.get("object_build_group_sha256") not in groups for b in registry["bindings"]):
        raise ValueError("source label lacks its native group")
    return registry



def verify_redundant_removal(candidate,diagnostics,recipe):
    """Recompute collision-only containment before existing-task dedup authority.

    Diagnostic shape flags select colliders; visual meshes cannot establish
    containment. Every retained collider must remain numerically identical.
    """
    import numpy as np
    from scipy.spatial import ConvexHull
    diagnostic=_json(_verify(diagnostics["builder_diagnostic"])["path"])
    _verify(diagnostics["builder_hulls"])
    rows=diagnostic.get("all_hull_stats",[])
    colliders={int(row["shape"]) for row in rows if row.get("collision_enabled") is True}
    removed=recipe.get("removed_collision_shapes",[]);containers=recipe.get("containers",{})
    if not colliders or not removed or len(set(removed))!=len(removed) or any(type(i) is not int for i in removed) or not set(removed)<colliders:
        raise ValueError("redundant repair needs explicit removed collision IDs and retained solids")
    if set(containers)!={str(i) for i in removed}:raise ValueError("each removed collider needs an explicit retained containing collider")
    with np.load(diagnostics["builder_hulls"]["path"],allow_pickle=False) as arrays:
        vertices={i:np.asarray(arrays[f"vertices_{i:04d}"],dtype=float) for i in colliders}
        faces={i:np.asarray(arrays[f"faces_{i:04d}"],dtype=int) for i in colliders}
    retained=colliders-set(removed)
    expected={"shapes":[{"kind":"mesh","vertices":vertices[i].tolist(),"faces":faces[i].tolist()} for i in sorted(retained)]}
    if digest(exact_surface(expected["shapes"]))!=digest(exact_surface(candidate["shapes"])):
        raise ValueError("union-preserving removal changed a retained collision component")
    records=[]
    for i in removed:
        points=vertices[i];container=containers[str(i)]
        if type(container) is not int or container not in retained:raise ValueError("containment proof used a visual/removed/nonexistent collider")
        if not np.isfinite(points).all() or np.linalg.matrix_rank(points-points[0],tol=1e-12)>2:
            raise ValueError("existing-task degenerate removal requires a planar component")
        hull=ConvexHull(vertices[container]);signed=points@hull.equations[:,:3].T+hull.equations[:,3]
        maximum=float(signed.max())
        if not maximum < -1e-9:raise ValueError("removed collision component is not strictly inside its retained convex solid")
        records.append({"removed_collision_shape":i,"container_collision_shape":container,"maximum_signed_halfspace_m":maximum})
    return {"schema":"hoi_collision_only_redundant_component_proof_v1","collision_shapes":sorted(colliders),"retained_shapes":sorted(retained),
        "removed":records,"retained_numeric_geometry_unchanged":True,"collision_union_preserved":True,"visuals_used":False,
        "inertia_claim":"none; preserve original explicit profile or derive missing moments from actual native components"}


def read_proposal(path):
    proposal=_read(path,PROPOSAL_SCHEMA)
    if proposal.get("status")!="review_only_not_approved_not_native_verified":raise ValueError("proposal cannot assert approval or native acceptance")
    for key in ("source_registry","original_build","original_fixture","candidate_geometry","review_proof"):_verify(proposal[key])
    diagnostics=proposal.get("diagnostics",{})
    if not {"builder_hulls","builder_diagnostic","failed_attempt","failed_source"}<=set(diagnostics):raise ValueError("repair needs original builder and failed-native lineage")
    for binding in diagnostics.values():_verify(binding)
    registry=_registry(proposal["source_registry"]["path"])
    groups={g["object_build_group_sha256"]:g for g in registry["object_cooking_groups"]}
    group=groups.get(proposal["original_group_sha256"])
    if group is None or proposal["original_build"]!=group["build_receipt"] or proposal["original_fixture"]!=group["cooking_fixture"]:
        raise ValueError("repair does not match the original registry build/fixture group")
    build=_json(proposal["original_build"]["path"]);fixture=_json(proposal["original_fixture"]["path"])
    from hoi.data.native_assets import BUILD_SCHEMA
    if build.get("schema")!=BUILD_SCHEMA or sha(build["native_asset"])!=build["native_asset_sha256"]:
        raise ValueError("repair original native input changed")
    if fixture.get("content_sha256")!=digest({k:v for k,v in fixture.items() if k!="content_sha256"}) or fixture["asset"]["build_receipt"]!=proposal["original_build"]:
        raise ValueError("repair original fixture identity mismatch")
    record=fixture["source_record"];geometry=_json(record["object_geometry_path"])
    if digest(geometry)!=record["object_geometry_sha256"] or record["object_geometry_sha256"]!=group["representative_geometry_sha256"]:
        raise ValueError("repair original label geometry identity mismatch")
    if build["object_geometry_sha256"]!=record["object_geometry_sha256"] or build["exact_surface_sha256"]!=digest(exact_surface(geometry["shapes"])):
        raise ValueError("repair original build is not the immutable label surface")
    if proposal.get("source_label_geometry_sha256")!=build["object_geometry_sha256"] or proposal.get("source_label_surface_sha256")!=build["exact_surface_sha256"]:
        raise ValueError("repair must preserve original label/source surface identities")
    candidate=_candidate(proposal["candidate_geometry"]["path"])
    if proposal["candidate_geometry"].get("geometry_sha256")!=digest(candidate) or proposal["candidate_geometry"].get("exact_surface_sha256")!=digest(exact_surface(candidate["shapes"])):
        raise ValueError("repair candidate geometry/surface identity mismatch")
    if proposal.get("recipe",{}).get("operation") not in OPERATIONS or proposal.get("cooking_contract")!=COOKING_CONTRACT:
        raise ValueError("unsupported explicit collision repair recipe/contract")
    mode=build.get("inertia_mode","explicit")
    if mode not in ("explicit","native_cooking_only_uniform_density") or proposal.get("source_inertia_mode")!=mode or proposal.get("preserved_physical_profile")!=build["physical_profile"]:
        raise ValueError("repair must preserve source inertia availability and existing complete profile")
    base={k:build["physical_profile"][k] for k in ("mass","friction","ground_friction")}
    if proposal.get("preserved_mass_material")!=base:raise ValueError("repair may not change source mass/material")
    if proposal["recipe"]["operation"]=="remove_contained_degenerate_component":
        if proposal.get("union_preserving_check")!=verify_redundant_removal(candidate,diagnostics,proposal["recipe"]):
            raise ValueError("stored collision-only containment proof differs from recomputation")
    return proposal,registry,build,fixture,candidate


def write_proposal(registry_path,group_sha,candidate_path,review_proof,diagnostics,recipe,output):
    """Make an immutable review document, never authorize or build an asset."""
    registry=_registry(registry_path);group=next(g for g in registry["object_cooking_groups"] if g["object_build_group_sha256"]==group_sha)
    build=_json(group["build_receipt"]["path"]);candidate=_candidate(candidate_path)
    proposal={"schema":PROPOSAL_SCHEMA,"status":"review_only_not_approved_not_native_verified",
        "source_registry":_identity(registry_path),"original_group_sha256":group_sha,
        "original_build":group["build_receipt"],"original_fixture":group["cooking_fixture"],
        "source_label_geometry_sha256":build["object_geometry_sha256"],"source_label_surface_sha256":build["exact_surface_sha256"],
        "candidate_geometry":{**_identity(candidate_path),"geometry_sha256":digest(candidate),"exact_surface_sha256":digest(exact_surface(candidate["shapes"]))},
        "review_proof":_identity(review_proof),"diagnostics":{k:_identity(v) for k,v in diagnostics.items()},
        "recipe":deepcopy(recipe),"cooking_contract":deepcopy(COOKING_CONTRACT),
        "preserved_mass_material":{k:build["physical_profile"][k] for k in ("mass","friction","ground_friction")},
        "source_inertia_mode":build.get("inertia_mode","explicit"),"preserved_physical_profile":deepcopy(build["physical_profile"]),
        "training_ready":False}
    if recipe.get("operation")=="remove_contained_degenerate_component":
        proposal["union_preserving_check"]=verify_redundant_removal(candidate,proposal["diagnostics"],recipe)
    proposal=_sealed(proposal)
    _write(output,proposal);read_proposal(output);return proposal


def require_approval(proposal_path,approval_path):
    proposal,*rest=read_proposal(proposal_path)
    if approval_path is None:raise ValueError("collision repair is pending explicit user approval")
    approval=_read(approval_path,APPROVAL_SCHEMA)
    if (approval.get("approved") is not True or approval.get("proposal_content_sha256")!=proposal["content_sha256"] or
        approval.get("candidate_sha256")!=proposal["candidate_geometry"]["sha256"] or set(approval.get("scopes",[]))!=set(required_scopes(proposal)) or
        not isinstance(approval.get("authorization_record"),str) or not approval["authorization_record"].strip()):
        raise ValueError("repair approval must explicitly bind this exact proposal/candidate and both scopes")
    authority=approval.get("authority")
    if authority=="existing_task_authorization_for_collision_union_preserving_dedup":
        if proposal["recipe"]["operation"]!="remove_contained_degenerate_component" or approval.get("containment_proof_sha256")!=proposal["review_proof"]["sha256"]:
            raise ValueError("existing-task authority applies only to the bound union-preserving containment proof")
    elif authority!="explicit_user_approval":
        raise ValueError("repair approval must state explicit-user or existing-task containment authority")
    return proposal,*rest


def read_binding(path):
    binding=_read(path,BINDING_SCHEMA)
    for key in ("proposal","approval","cooking_input_build"):_verify(binding[key])
    proposal,registry,original,fixture,candidate=require_approval(binding["proposal"]["path"],binding["approval"]["path"])
    build=_json(binding["cooking_input_build"]["path"])
    from hoi.data.native_assets import BUILD_SCHEMA
    if (build.get("schema")!=BUILD_SCHEMA or build.get("object_geometry_sha256")!=digest(candidate) or
        build.get("exact_surface_sha256")!=digest(exact_surface(candidate["shapes"])) or
        build.get("physical_profile")!=proposal["preserved_physical_profile"] or build.get("inertia_mode","explicit")!=proposal["source_inertia_mode"] or
        build.get("collision_approximation")!="convexHull" or build.get("hull_vertex_policy")!=COOKING_CONTRACT["max_hull_vertices"] or
        build.get("training_ready") is not False or sha(build["native_asset"])!=build["native_asset_sha256"]):
        raise ValueError("repair cooking input does not match the approved candidate and physical contract")
    if (build.get("physical_evidence",{}).get("collision_repair_proposal")!=binding["proposal"] or
        build.get("physical_evidence",{}).get("collision_repair_approval")!=binding["approval"]):
        raise ValueError("repair cooking build lost its exact approval/proposal evidence")
    expected=digest({"original_group_sha256":proposal["original_group_sha256"],"proposal_sha256":proposal["content_sha256"],
                     "approval_sha256":binding["approval"]["sha256"],"cooking_input_build_sha256":binding["cooking_input_build"]["sha256"]})
    if binding.get("object_build_group_sha256")!=expected:raise ValueError("repair-aware native build group identity mismatch")
    return binding,proposal,registry,original,fixture,build,candidate


def write_approved_sources(proposal_path,approval_path,output):
    """Fresh repair input and fixture. No cooking/physics/GPU jobs are run here."""
    proposal,registry,original,fixture,candidate=require_approval(proposal_path,approval_path)
    from hoi.data.native_assets import write_native_source
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    evidence={**deepcopy(original["physical_evidence"]),"collision_repair_proposal":_identity(proposal_path),
              "collision_repair_approval":_identity(approval_path),"inertia_policy":"preserve_explicit_source_profile_else_derive_missing_from_actual_closed_components"}
    if proposal["source_inertia_mode"]!="explicit":
        evidence["inertia_derivation"]={"status":"waiting_for_actual_repaired_native_closed_convex_components"}
    build=write_native_source(candidate,output/"input",physical_profile=proposal["preserved_physical_profile"],physical_evidence=evidence,
        collision_approximation="convexHull",inertia_mode=proposal["source_inertia_mode"],preserve_hull_vertices=True)
    binding={"schema":BINDING_SCHEMA,"proposal":_identity(proposal_path),"approval":_identity(approval_path),
             "cooking_input_build":_identity(output/"input/build.json"),"training_ready":False}
    binding["object_build_group_sha256"]=digest({"original_group_sha256":proposal["original_group_sha256"],"proposal_sha256":proposal["content_sha256"],
        "approval_sha256":binding["approval"]["sha256"],"cooking_input_build_sha256":binding["cooking_input_build"]["sha256"]})
    binding=_sealed(binding);_write(output/"repair_binding.json",binding)
    result=deepcopy(fixture);result.pop("content_sha256",None);result["schema"]=FIXTURE_SCHEMA
    result["repair_binding"]=_identity(output/"repair_binding.json")
    asset=result["asset"];asset.update(native_asset=build["native_asset"],native_asset_sha256=build["native_asset_sha256"],
        physical_profile=build["physical_profile"],build_receipt=binding["cooking_input_build"],
        collision_hash=digest({"approved_repair":binding["content_sha256"],"input_surface":build["exact_surface_sha256"]}),
        hull_vertex_policy=build["hull_vertex_policy"],repair_binding=result["repair_binding"])
    if "inertia_mode" in build:asset["inertia_mode"]=build["inertia_mode"]
    else:asset.pop("inertia_mode",None)
    # asset.object_geometry_sha256 and source_record intentionally remain the
    # original label identity; schema v2 declares the separate actual input.
    result=_sealed(result);_write(output/"cooking_fixture.json",result);read_binding(output/"repair_binding.json")
    return result


def fixture_repair_context(fixture):
    """Validated v2 alternative; v1 validators keep their exact-equality rule."""
    if fixture.get("schema")!=FIXTURE_SCHEMA or fixture.get("content_sha256")!=digest({k:v for k,v in fixture.items() if k!="content_sha256"}):raise ValueError("not an intact explicit repair fixture")
    _verify(fixture["repair_binding"])
    values=read_binding(fixture["repair_binding"]["path"])
    binding,proposal,registry,original,old_fixture,build,candidate=values
    if fixture["asset"].get("repair_binding")!=fixture["repair_binding"] or fixture["asset"]["build_receipt"]!=binding["cooking_input_build"]:
        raise ValueError("repair fixture input binding mismatch")
    if fixture["source_record"]!=old_fixture["source_record"] or any(fixture[k]!=old_fixture[k] for k in ("native_robot_asset","reference_kinematics","robot_geometry_sha256","robot_recipe")):
        raise ValueError("repair fixture may not alter reference, robot or scene provenance")
    for key in ("native_asset","native_asset_sha256","physical_profile","inertia_mode","hull_vertex_policy"):
        if fixture["asset"].get(key)!=build.get(key):raise ValueError("repair fixture changed its approved cooking input: "+key)
    for key in ("asset_id","object_asset","exact_scale","mesh_to_object"):
        if fixture["asset"].get(key)!=old_fixture["asset"].get(key):raise ValueError("repair fixture changed source frame/scale/identity: "+key)
    if fixture["asset"].get("object_geometry_sha256")!=proposal["source_label_geometry_sha256"]:
        raise ValueError("repair fixture changed the immutable label geometry identity")
    return values


def derive_registry(registry_path,repair_fixture_paths,output):
    registry=_registry(registry_path)
    if registry.get("repair_derivation"):raise ValueError("derive repairs directly from the immutable original registry")
    result=deepcopy(registry);result.pop("content_sha256",None);groups={g["object_build_group_sha256"]:g for g in result["object_cooking_groups"]}
    replacements={};receipts=[]
    for fixture_path in repair_fixture_paths:
        fixture=_json(fixture_path)
        if fixture.get("content_sha256")!=digest({k:v for k,v in fixture.items() if k!="content_sha256"}):raise ValueError("repair fixture hash changed")
        binding,proposal,_,_,_,build,_=fixture_repair_context(fixture)
        if proposal["source_registry"]!=_identity(registry_path):raise ValueError("repair belongs to another original registry")
        old=proposal["original_group_sha256"];new=binding["object_build_group_sha256"]
        if old not in groups or old in replacements:raise ValueError("unknown/duplicate repaired registry group")
        group=groups[old];group.update(object_build_group_sha256=new,build_receipt=binding["cooking_input_build"],
            cooking_fixture=_identity(fixture_path),inertia_mode=build.get("inertia_mode","explicit"),repair_binding=fixture["repair_binding"],original_group_sha256=old)
        replacements[old]=new;receipts.append(fixture["repair_binding"])
    if not replacements:raise ValueError("derived registry requires explicit approved repairs")
    for record in result["bindings"]:
        if record.get("object_build_group_sha256") in replacements:record["object_build_group_sha256"]=replacements[record["object_build_group_sha256"]]
    result["repair_derivation"]={"original_registry":_identity(registry_path),"group_replacements":replacements,"approved_repairs":receipts,
        "label_geometry_count":len(registry["bindings"]),"source_labels_and_split_unchanged":True}
    result=_sealed(result);validate_derived_registry(result);_write(output,result);return result


def validate_derived_registry(registry):
    derivation=registry.get("repair_derivation")
    if derivation is None:return
    _verify(derivation["original_registry"]);original=_registry(derivation["original_registry"]["path"])
    replacements=derivation["group_replacements"];proofs={}
    for identity in derivation["approved_repairs"]:
        _verify(identity);binding,proposal,*_=read_binding(identity["path"])
        if proposal["source_registry"]!=derivation["original_registry"]:raise ValueError("derived repair source registry mismatch")
        old=proposal["original_group_sha256"]
        if old in proofs or replacements.get(old)!=binding["object_build_group_sha256"]:raise ValueError("ambiguous repaired group replacement")
        proofs[old]=(identity,binding)
    if set(proofs)!=set(replacements):raise ValueError("derived group replacement lacks approval")
    expected=deepcopy(original);expected.pop("content_sha256",None)
    actual=deepcopy(registry);actual.pop("content_sha256",None);actual.pop("repair_derivation",None)
    current_groups={g["object_build_group_sha256"]:g for g in actual["object_cooking_groups"]}
    if len(current_groups)!=len(actual["object_cooking_groups"]):raise ValueError("duplicate derived native groups")
    for group in expected["object_cooking_groups"]:
        old=group["object_build_group_sha256"]
        if old not in replacements:continue
        identity,binding=proofs[old];new=replacements[old];actual_group=current_groups.get(new,{})
        _verify(actual_group["cooking_fixture"]);actual_context=fixture_repair_context(_json(actual_group["cooking_fixture"]["path"]))
        group.update(object_build_group_sha256=new,build_receipt=binding["cooking_input_build"],cooking_fixture=actual_group["cooking_fixture"],
            inertia_mode=actual_context[5].get("inertia_mode","explicit"),repair_binding=identity,original_group_sha256=old)
    for record in expected["bindings"]:
        if record.get("object_build_group_sha256") in replacements:record["object_build_group_sha256"]=replacements[record["object_build_group_sha256"]]
    if actual!=expected:raise ValueError("derived registry changed original labels, membership, split, or an unrepaired input")


def apply_hull_vertex_contract(prim,count):
    """Official per-shape Newton import option; repair-only explicit hulls."""
    from pxr import Sdf
    if type(count) is not int or count<4:raise ValueError("explicit convex repair hull needs at least four vertices")
    prim.AddAppliedSchema("NewtonMeshCollisionAPI")
    prim.CreateAttribute("newton:maxHullVertices",Sdf.ValueTypeNames.Int).Set(count)



def compare_candidate_export(binding_path,meshes):
    """Pair every actual convex solid with its approved input; no sampled proxy."""
    import numpy as np
    from scipy.spatial import ConvexHull
    from scipy.optimize import linear_sum_assignment
    _,proposal,_,_,_,_,candidate=read_binding(binding_path)
    inputs=candidate["shapes"]
    if len(inputs)!=len(meshes):raise ValueError("native repair import changed the approved component count")
    a=[np.asarray(m["vertices"],dtype=float) for m in inputs]
    b=[np.asarray(m["vertices"],dtype=float) for m in meshes]
    ah=[ConvexHull(v) for v in a];bh=[ConvexHull(v) for v in b]
    def outside(v,h):return max(0.,float((v@h.equations[:,:3].T+h.equations[:,3]).max()))
    cost=np.asarray([[max(outside(x,hy),outside(y,hx)) for y,hy in zip(b,bh)] for x,hx in zip(a,ah)])
    rows,cols=linear_sum_assignment(cost);tol=COOKING_CONTRACT["native_reimport_tolerance_m"]
    error=float(cost[rows,cols].max())
    if not np.isfinite(cost).all() or error>tol:raise ValueError("native repair import changed an approved convex solid beyond bound tolerance")
    widths_a=[float(np.ptp(v@h.equations[:,:3].T,axis=0).min()) for v,h in zip(a,ah)]
    widths_b=[float(np.ptp(v@h.equations[:,:3].T,axis=0).min()) for v,h in zip(b,bh)]
    volume_error=[abs(float(bh[j].volume/ah[i].volume)-1.) for i,j in zip(rows,cols)]
    width_error=[abs(widths_b[j]/widths_a[i]-1.) for i,j in zip(rows,cols)]
    if max(volume_error)>COOKING_CONTRACT["component_volume_relative_tolerance"] or max(width_error)>COOKING_CONTRACT["component_minimum_facet_width_relative_tolerance"]:
        raise ValueError("native repair import changed component volume or thin-component width beyond relative bound")
    return {"schema":"hoi_approved_repair_native_convex_comparison_v1","method":"bijective_components_bidirectional_all_vertex_normalized_halfspace_violation",
        "candidate_sha256":proposal["candidate_geometry"]["sha256"],"native_components":len(meshes),"tolerance_m":tol,
        "maximum_plane_violation_m":error,
        "component_volume_relative_tolerance":COOKING_CONTRACT["component_volume_relative_tolerance"],
        "component_minimum_facet_width_relative_tolerance":COOKING_CONTRACT["component_minimum_facet_width_relative_tolerance"],
        "maximum_component_volume_relative_error":max(volume_error),"maximum_component_minimum_facet_width_relative_error":max(width_error),
        "pairs":[{"input_component":int(i),"native_component":int(j),"maximum_plane_violation_m":float(cost[i,j]),
        "input_volume_m3":float(ah[i].volume),"native_volume_m3":float(bh[j].volume),
        "input_minimum_facet_width_m":widths_a[i],"native_minimum_facet_width_m":widths_b[j]} for i,j in zip(rows,cols)],"passed":True}


def dependency_bindings(path):
    binding,proposal,_,original,_,build,_=read_binding(path)
    values=[_identity(path),binding["proposal"],binding["approval"],binding["cooking_input_build"],
            proposal["source_registry"],proposal["original_build"],proposal["original_fixture"],
            proposal["candidate_geometry"],proposal["review_proof"],*proposal["diagnostics"].values(),
            {"path":original["native_asset"],"sha256":original["native_asset_sha256"]},
            {"path":build["native_asset"],"sha256":build["native_asset_sha256"]}]
    return [{"path":v["path"],"sha256":v["sha256"]} for v in values]


def validate_final_entry(entry):
    identity=entry.get("repair_binding")
    if identity is None:return
    _verify(identity);binding,proposal,registry,original,_,_,_=read_binding(identity["path"])
    group=next(g for g in registry["object_cooking_groups"] if g["object_build_group_sha256"]==proposal["original_group_sha256"])
    if (entry.get("object_geometry_sha256") not in group["members"] or entry.get("label_surface_sha256")!=original["exact_surface_sha256"] or
        entry.get("source_build_receipt")!=proposal["original_build"] or entry.get("cooking_input_build_receipt")!=binding["cooking_input_build"] or
        entry.get("inertia_mode") is not None or entry.get("hull_vertex_policy")!=COOKING_CONTRACT["max_hull_vertices"]):
        raise ValueError("final repaired asset lost approved source/input lineage")
    for key in ("source_build_receipt","cooking_input_build_receipt","compiled_recipe_receipt","final_native_build_receipt"):_verify(entry[key])
    from hoi.data.native_assets import uniform_density_profile
    if proposal["source_inertia_mode"]=="explicit":
        actual=proposal["preserved_physical_profile"]
        if entry["physical_profile"]!=actual:raise ValueError("repair changed explicit source COM/inertia profile")
    else:
        actual,_=uniform_density_profile({"shapes":[dict(m,kind="mesh") for m in entry["collision_meshes"]]},**proposal["preserved_mass_material"])
    import numpy as np
    for key in ("mass","friction","ground_friction","com","inertia"):
        if not np.allclose(actual[key],entry["physical_profile"][key],atol=1e-12,rtol=1e-10):
            raise ValueError("final repaired COM/inertia/material differs from actual compiled component rule")
    final=_json(entry["final_native_build_receipt"]["path"])
    if final.get("native_asset_sha256")!=entry["native_asset_sha256"] or final.get("physical_profile")!=entry["physical_profile"] or final.get("hull_vertex_policy")!=COOKING_CONTRACT["max_hull_vertices"]:
        raise ValueError("final repaired USD build lost its actual profile/hull preservation contract")


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("proposal")
    for name in ("registry","group","candidate","review-proof","diagnostics-json","recipe-json","output"):p.add_argument("--"+name,required=True)
    p=sub.add_parser("build")
    for name in ("proposal","approval","output"):p.add_argument("--"+name,required=True)
    p=sub.add_parser("derive-registry");p.add_argument("--registry",required=True);p.add_argument("--repair-fixture",action="append",required=True);p.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    if args.command=="proposal":result=write_proposal(args.registry,args.group,args.candidate,args.review_proof,_json(args.diagnostics_json),_json(args.recipe_json),args.output)
    elif args.command=="build":result=write_approved_sources(args.proposal,args.approval,args.output)
    else:result=derive_registry(args.registry,args.repair_fixture,args.output)
    print(json.dumps({"schema":result["schema"],"content_sha256":result["content_sha256"],"training_ready":False}))


if __name__=="__main__":main()
