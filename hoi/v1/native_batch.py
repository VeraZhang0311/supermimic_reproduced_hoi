"""Resumable, partition-bound native acceptance; no training or job submission.

Each partition/asset is a fresh native process with 8 worlds, 64 smoke control
intervals and all 31 pair probes. Cached compiled code may be shared; evidence
and bundle identities never cross partitions. Slurm schedules at most 4 slots.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from hoi.data.native_assets import digest, sha
from hoi.data.native_cooking import (_identity, verify_identity, _json, _write,
    _source_files, _runtime_versions, _task_lock, _planning_identity_cache)

PLAN_SCHEMA="hoi_native_acceptance_batch_plan_v1"
SUCCESS_SCHEMA="hoi_native_acceptance_task_success_v1"
SCHEDULE_SCHEMA="hoi_native_acceptance_schedule_v1"
PARTITIONS=("train","protected_eval")
SMOKE_PHASE="first_existing_smoke_control_interval_after_probes_before_autoreset"


def _seal(value):
    return {**value,"content_sha256":digest(value)}


def _read_sealed(path,schema):
    value=_json(path)
    if value.get("schema")!=schema or value.get("content_sha256")!=digest({k:v for k,v in value.items() if k!="content_sha256"}):
        raise ValueError("batch identity/schema mismatch: "+str(path))
    return value


def _bound_files(value):
    result=[]
    if isinstance(value,dict):
        if isinstance(value.get("path"),str) and isinstance(value.get("sha256"),str):
            result.append(_identity(value["path"],value["sha256"]))
        for child in value.values():result.extend(_bound_files(child))
    elif isinstance(value,list):
        for child in value:result.extend(_bound_files(child))
    return result


def _task_inputs(bundle,asset,clips):
    values=[bundle[k] for k in ("manifest","recipes","native_robot_asset","reference_kinematics","hands")]
    if "prepared_geometry" in bundle:values.append(bundle["prepared_geometry"])
    values.extend([asset,*clips]);bindings=_bound_files(values)
    if asset.get("repair_binding"):
        from hoi.data.native_repairs import dependency_bindings
        bindings.extend(verify_identity(x) for x in dependency_bindings(asset["repair_binding"]["path"]))
    bindings.append(_identity(asset["native_asset"],asset["native_asset_sha256"]))
    if "robot_fk" in bundle:
        bindings.append(_identity(bundle["robot_fk"]["mjcf_path"],bundle["robot_fk"]["mjcf_sha256"]))
    for clip in clips:
        root=Path(clip["resolved_source_output"])
        bindings.extend((_identity(root/"metadata.json",clip["metadata_sha256"]),_identity(root/"labels.npz",clip["labels_sha256"])))
    unique={b["path"]:b for b in bindings}
    if len({(b["path"],b["sha256"]) for b in bindings})!=len(unique):
        raise ValueError("same input path has conflicting SHA256 bindings")
    return list(unique.values())


def _native_args(values):
    if not isinstance(values,list) or not all(isinstance(x,str) for x in values):raise ValueError("native arguments must be a JSON list of strings")
    reserved=("--output","--asset-id","--num-envs","--steps","--seed","--bundle","--legacy-prepared-fixture","--curated-cooking-fixture","--robot-asset-evidence")
    if any(any(x==key or x.startswith(key+"=") for key in reserved) for x in values):raise ValueError("batch owns native input/output/count/seed arguments")
    if not all(x in values for x in ("--hand-collision","--self-collision")) or any(x in values for x in ("--no-hand-collision","--no-self-collision")):
        raise ValueError("native batch requires explicit hand/self collision ON")
    if values.count("--capture-smoke-visual")!=1:
        raise ValueError("full native batch requires exactly one --capture-smoke-visual")
    return values


@_planning_identity_cache()
def make_plan(bundle_paths,source,python,native_args,output,*,seed=20261031):
    from hoi.data.curated_bundle import read_curated_bundle_header
    if len(bundle_paths)!=2:raise ValueError("full batch requires exactly train and protected_eval bundles")
    source=Path(source).resolve();python=Path(python).absolute();output=Path(output).resolve()
    native_args=_native_args(native_args);bundles={};tasks=[]
    for path in bundle_paths:
        bundle=read_curated_bundle_header(path);partition=bundle["partition"]
        if partition not in PARTITIONS or partition in bundles:raise ValueError("one distinct full bundle per partition required")
        if bundle["scope"].get("is_subset") or bundle["scope"].get("purpose")!="curated_reference_bundle":
            raise ValueError("pilot/subset bundles cannot supply full native acceptance")
        bundles[partition]={**_identity(path),"bundle_content_sha256":bundle["bundle_content_sha256"]}
        assets={int(a["asset_id"]):a for a in bundle["assets"]}
        for aid in sorted({int(c["asset_id"]) for c in bundle["clips"]}):
            clips=[c for c in bundle["clips"] if int(c["asset_id"])==aid]
            task={"task_id":f"{partition}_asset_{aid:04d}","partition":partition,"asset_id":aid,"asset":assets[aid],
                  "clip_names":[c["name"] for c in clips],"inputs":_task_inputs(bundle,assets[aid],clips)}
            task["task_sha256"]=digest(task);tasks.append(task)
    tasks.sort(key=lambda t:(PARTITIONS.index(t["partition"]),t["asset_id"]))
    if not tasks or set(bundles)!=set(PARTITIONS):raise ValueError("full native batch has no executable tasks")
    execution={"source":str(source),"source_files":_source_files(source),"python":str(python),"python_identity":_identity(python),
               "driver_identity":_identity(__file__),"runtime_versions":_runtime_versions(python),"native_args":native_args,
               "num_envs":8,"steps":64,"pair_probe_cases":31,"pair_probe_ticks":8,"seed":int(seed),"maximum_concurrent_slots":4}
    plan=_seal({"schema":PLAN_SCHEMA,"bundles":bundles,"execution":execution,"execution_sha256":digest(execution),"tasks":tasks,
                "output":str(output),"task_counts":{p:sum(t["partition"]==p for t in tasks) for p in PARTITIONS},
                "training_started":False,"scope":"independent actual native evidence for every partition/asset; no cross-bundle reuse"})
    output.mkdir(parents=True,exist_ok=False);_write(output/"plan.json",plan)
    return plan


def read_plan(path):
    plan=_read_sealed(path,PLAN_SCHEMA);execution=plan["execution"]
    if plan["execution_sha256"]!=digest(execution):raise ValueError("execution identity mismatch")
    if (execution["num_envs"],execution["steps"],execution["pair_probe_cases"],execution["pair_probe_ticks"],execution["maximum_concurrent_slots"])!=(8,64,31,8,4):
        raise ValueError("native acceptance batch contract changed")
    _native_args(execution["native_args"])
    verify_identity(execution["python_identity"]);verify_identity(execution["driver_identity"])
    if _source_files(execution["source"])!=execution["source_files"]:raise ValueError("frozen executable source changed")
    if _runtime_versions(execution["python"])!=execution["runtime_versions"]:raise ValueError("native runtime versions changed")
    for binding in plan["bundles"].values():verify_identity(binding)
    ids=[]
    for task in plan["tasks"]:
        if task["task_sha256"]!=digest({k:v for k,v in task.items() if k!="task_sha256"}):raise ValueError("task identity changed")
        ids.append(task["task_id"])
    if len(set(ids))!=len(ids):raise ValueError("duplicate task IDs")
    return plan


def _visual_artifacts(plan,report):
    """Verify saved observation/render identities without running rendering or physics."""
    visual=report.get("smoke_visual_snapshot",{})
    if (visual.get("enabled") is not True or visual.get("captured") is not True or
            visual.get("extra_physics_steps")!=0 or visual.get("phase")!=SMOKE_PHASE or
            visual.get("captured_existing_control_interval")!=1):
        raise ValueError("actual first-interval smoke visual capture is missing or changed")
    native_root=Path(report["compiled_geometry_export"]).resolve().parent.parent
    root=native_root/"smoke_visual"
    manifest_path=Path(visual["manifest"]).resolve()
    rendering_path=Path(visual["rendering_report"]).resolve()
    if manifest_path!=root/"compiled_model_manifest.json" or rendering_path!=root/"render/rendering.json":
        raise ValueError("smoke visual artifacts are outside this native attempt")
    artifacts={"smoke_manifest":_identity(manifest_path,visual["manifest_sha256"])}
    manifest=_json(manifest_path);scene=manifest.get("scene_snapshot",{})
    if (manifest.get("schema")!="hoi_actual_native_model_dump_v1" or manifest.get("physics_steps")!=0 or
            scene.get("schema")!="hoi_actual_native_scene_snapshot_v1" or scene.get("num_envs")!=8 or
            scene.get("capture_phase")!=SMOKE_PHASE):
        raise ValueError("smoke visual manifest does not describe the actual eight-world capture")
    mapping=manifest.get("contact_mapping",{})
    if not mapping or {k:v for k,v in mapping.items() if k!="geom_environment_ids"}!=report.get("contact"):
        raise ValueError("smoke visual contact mapping differs from the actual native recorder")
    environments=mapping.get("geom_environment_ids")
    if (not isinstance(environments,list) or len(environments)!=len(mapping.get("geom_names",[])) or
            any(type(e) is not int or not -1<=e<8 for e in environments)):
        raise ValueError("smoke visual geom environment identities are missing")
    declared=report.get("native_scene_supports",{}).get("declared_support_ids")
    if scene.get("declared_support_ids")!=declared:
        raise ValueError("smoke visual declared supports differ from native support evidence")
    binary=manifest["binary"];live=manifest["live_gpu_arrays"]
    if binary.get("collision_arrays_exact_roundtrip") is not True:
        raise ValueError("smoke visual MJB lacks the actual collision-array roundtrip")
    for key,binding,name in (("smoke_binary",binary,"compiled_model.mjb"),("smoke_live_arrays",live,"compiled_model_live_arrays.pt")):
        if Path(binding["path"]).resolve()!=root/name:
            raise ValueError("smoke visual bound data is outside this native capture")
        artifacts[key]=verify_identity(binding)
    rendering=_json(rendering_path)
    if (rendering.get("schema")!="hoi_actual_native_snapshot_render_v1" or rendering.get("environment")!=0 or
            type(rendering.get("saved_frame")) is not int or rendering["saved_frame"]<0 or
            any(rendering.get(k)!=0 for k in ("physics_steps","fk_calls","policy_calls"))):
        raise ValueError("smoke rendering is missing its saved-frame zero-physics/FK/policy contract")
    for key,artifact in (("manifest_sha256","smoke_manifest"),("binary_sha256","smoke_binary"),("live_arrays_sha256","smoke_live_arrays")):
        if rendering.get(key)!=artifacts[artifact]["sha256"]:
            raise ValueError("smoke rendering uses a different captured "+key)
    renderer_sha=plan["execution"]["source_files"].get("hoi/v1/native_snapshot_render.py")
    if not renderer_sha or rendering.get("renderer_sha256")!=renderer_sha:
        raise ValueError("smoke renderer differs from the planned frozen source")
    visible=rendering.get("visible_support_ids");errors=rendering.get("support_geom_pose_errors")
    if (not isinstance(visible,list) or not isinstance(errors,dict) or not isinstance(declared,list) or
            len(set(visible))!=len(visible) or not set(visible).issubset(declared) or set(errors)!=set(declared)):
        raise ValueError("smoke rendered support identities differ from the captured scene")
    for values in errors.values():
        if not isinstance(values,dict) or any(type(values.get(k)) not in (float,int) or not math.isfinite(values[k]) or
                not 0<=values[k]<=2e-5 for k in ("position_error_m","rotation_matrix_error")):
            raise ValueError("smoke rendered support transforms do not match saved actual poses")
    artifacts["smoke_rendering"]=_identity(rendering_path)
    artifacts["smoke_preview"]=_identity(rendering_path.parent/"preview.png",rendering["preview_sha256"])
    return artifacts


def _scene_reset_artifacts(plan,task,report):
    """Check optional terminal reset evidence; this does not perform resets."""
    if "--audit-scene-resets" not in plan["execution"]["native_args"]:
        return {}
    audit=report.get("native_scene_reset_audit",{})
    if (audit.get("schema")!="hoi_native_terminal_scene_reset_audit_v1" or audit.get("enabled") is not True or
            audit.get("passed") is not True or audit.get("phase")!="terminal_auxiliary_after_smoke_state_discarded" or
            audit.get("selection_rule")!="every_asset_clip_first_middle_last_and_presence_change_neighbors_v1" or
            audit.get("asset_id")!=task["asset_id"] or audit.get("num_envs")!=8 or
            audit.get("reset_noise") is not False or audit.get("state_discarded") is not True):
        raise ValueError("requested terminal scene reset audit is missing or has a different scope")
    for key in ("additional_physics_steps","attempted_advance_calls","attempted_sim_step_calls","attempted_solver_step_calls","presence_mismatches"):
        if type(audit.get(key)) is not int or audit[key]!=0:
            raise ValueError("terminal scene reset audit added simulation work or mismatched presence")
    for prefix in ("physics_steps","recorder_substeps"):
        before,after=audit.get(prefix+"_before"),audit.get(prefix+"_after")
        if type(before) is not int or before<0 or type(after) is not int or before!=after:
            raise ValueError("terminal scene reset audit changed the actual "+prefix+" counter")
    errors=audit.get("maximum_errors",{})
    if any(type(errors.get(k)) not in (float,int) or not math.isfinite(errors[k]) or not 0<=errors[k]<=2e-5
            for k in ("position_m","quaternion","linear_velocity_m_s","angular_velocity_rad_s")):
        raise ValueError("terminal scene reset observed poses/velocities differ from bound references")
    clips=audit.get("clips");names=audit.get("clip_names")
    if (not isinstance(names,list) or any(not isinstance(n,str) for n in names) or sorted(names)!=sorted(task["clip_names"]) or
            not isinstance(clips,list) or any(not isinstance(c,dict) or not isinstance(c.get("name"),str) for c in clips) or
            sorted(c["name"] for c in clips)!=sorted(task["clip_names"])):
        raise ValueError("terminal scene reset audit does not cover every planned asset clip exactly once")
    support_bindings={c["name"]:c.get("bindings",[]) for c in report["native_scene_supports"]["clips"]}
    requested=0;expected_samples=[];offset=0
    by_name={c["name"]:c for c in clips}
    if audit.get("declared_support_ids")!=report["native_scene_supports"].get("declared_support_ids") or audit.get("formal_training_started") is not False:
        raise ValueError("terminal scene reset declared support/training scope differs")
    for index,name in enumerate(task["clip_names"]):
        clip=by_name[name]
        if clip.get("clip_index")!=index:
            raise ValueError("terminal scene reset clip index differs from the selected motion bank")
        n=clip.get("state_frames");selected=clip.get("selected_local_frames");changes=clip.get("presence_change_frames")
        if (type(n) is not int or n<1 or not isinstance(selected,list) or any(type(i) is not int or not 0<=i<n for i in selected) or
                selected!=sorted(set(selected)) or not isinstance(changes,list) or any(type(i) is not int or not 1<=i<n for i in changes) or
                changes!=sorted(set(changes))):
            raise ValueError("terminal scene reset selected frame identities are invalid")
        required={0,n//2,n-1,*changes,*(i-1 for i in changes)}
        if selected!=sorted(required):
            raise ValueError("terminal scene reset selection omits or changes required representative/boundary frames")
        if clip.get("bindings")!=support_bindings[clip["name"]]:
            raise ValueError("terminal scene reset clip bindings differ from native support provenance")
        expected_samples.extend((name,f,index,offset+f) for f in selected)
        requested+=len(selected);offset+=n
    if (type(audit.get("requested_sample_count")) is not int or audit["requested_sample_count"]!=requested or
            type(audit.get("completed_sample_count")) is not int or audit["completed_sample_count"]!=requested or
            type(audit.get("reset_batches")) is not int or audit["reset_batches"]!=(requested+7)//8):
        raise ValueError("terminal scene reset sample coverage/count is incomplete")
    binding=verify_identity(audit["artifact"])
    native_root=Path(report["compiled_geometry_export"]).resolve().parent.parent
    if Path(binding["path"])!=native_root/"scene_reset_audit.json":
        raise ValueError("terminal scene reset audit artifact belongs to another native attempt")
    detail=_json(binding["path"]);samples=detail.get("samples")
    if ({k:v for k,v in detail.items() if k not in ("samples","artifact")}!={k:v for k,v in audit.items() if k!="artifact"} or
            not isinstance(samples,list) or len(samples)!=requested or any(not isinstance(row,dict) for row in samples)):
        raise ValueError("terminal scene reset detailed artifact differs from its report or sample count")
    observed_samples=[(r.get("clip_name"),r.get("local_frame"),r.get("clip_index"),r.get("frame")) for r in samples]
    if observed_samples!=expected_samples or any(type(r.get("environment_id")) is not int or r["environment_id"]!=i%8 for i,r in enumerate(samples)):
        raise ValueError("terminal scene reset detailed samples omit or replace planned clip/frame/environment identities")
    return {"scene_reset_audit":binding}


def _validate_report(plan,task,report):
    from .gates import validate_native_evidence
    from .native_input_contract import SCHEMA as INPUT_SCHEMA
    execution=plan["execution"]
    if (report.get("asset_id"),report.get("num_envs"),report.get("steps"),report.get("seed"))!=(task["asset_id"],8,64,execution["seed"]):
        raise ValueError("native report asset/env/step/seed differs from full batch plan")
    if report.get("formal_training_started") is not False or report.get("source_code_sha256")!=execution["source_files"]:
        raise ValueError("native source/training scope mismatch")
    if report.get("native_runtime",{}).get("versions")!=execution["runtime_versions"]:raise ValueError("report native runtime differs from planned packages")
    probes=[v for v in report.get("pair_probes",{}).values() if isinstance(v,dict) and "ticks" in v]
    if len(probes)!=31 or any(p["ticks"]!=8 for p in probes):raise ValueError("full native batch requires all 31 eight-tick probes")
    inp=report.get("native_input_contract",{})
    if inp.get("schema")!=INPUT_SCHEMA or inp.get("passed") is not True or inp.get("cuda_verified") is not True or inp.get("num_envs")!=8:
        raise ValueError("actual CUDA transition input contract is missing")
    if any(inp.get(k)!=0 for k in ("normalizer_updates","optimizer_steps","additional_physics_steps")) or inp.get("rng_preserved") is not True or inp.get("outputs_applied_to_simulator") is not False or inp.get("untrained_validation_only") is not True:
        raise ValueError("input probe changed learning/collection state")
    verify_identity(inp["artifact"])
    gate=validate_native_evidence(report,expected_physics={"hand_collision":True,"self_collision":True},required_asset_ids=[task["asset_id"]],
        expected_bundle_sha256=plan["bundles"][task["partition"]]["bundle_content_sha256"],expected_asset_profiles={task["asset_id"]:task["asset"]["physical_profile"]})
    if not gate["passed"]:raise ValueError("native gate failed: "+"; ".join(gate["blockers"]))
    clips=report.get("native_scene_supports",{}).get("clips")
    if (not isinstance(clips,list) or any(not isinstance(c,dict) or not isinstance(c.get("name"),str) for c in clips) or
            sorted(c["name"] for c in clips)!=sorted(task["clip_names"])):
        raise ValueError("native support clips do not cover every planned asset clip exactly once")
    _visual_artifacts(plan,report)
    _scene_reset_artifacts(plan,task,report)
    return gate


def _success(plan,task,attempt):
    receipt=_read_sealed(attempt/"SUCCESS.json",SUCCESS_SCHEMA)
    if receipt.get("task_sha256")!=task["task_sha256"] or receipt.get("plan_sha256")!=plan["content_sha256"]:
        raise ValueError("success receipt belongs to a different plan/task")
    for binding in receipt["artifacts"].values():verify_identity(binding)
    report=_json(receipt["artifacts"]["native"]["path"])
    gate=_validate_report(plan,task,report)
    for key,binding in {**_visual_artifacts(plan,report),**_scene_reset_artifacts(plan,task,report)}.items():
        if receipt["artifacts"].get(key)!=binding:
            raise ValueError("success receipt does not bind the validated auxiliary artifact: "+key)
    if _json(receipt["artifacts"]["gate"]["path"])!=gate:raise ValueError("saved gate differs from recomputed gate")
    return receipt


def _completed(plan,task):
    root=Path(plan["output"])/"tasks"/task["task_id"]
    found=[]
    for attempt in sorted(root.glob("attempt_*")):
        if (attempt/"SUCCESS.json").exists():found.append((attempt,_success(plan,task,attempt)))
    if len(found)>1:raise ValueError("multiple successful producers for "+task["task_id"])
    return found[0] if found else None


@_planning_identity_cache()
def status(plan):
    result=[]
    for index,task in enumerate(plan["tasks"]):
        try:
            for binding in task["inputs"]:verify_identity(binding)
            success=_completed(plan,task)
            attempts=list((Path(plan["output"])/"tasks"/task["task_id"]).glob("attempt_*"))
            state="complete" if success else ("failed_or_interrupted" if attempts else "pending")
            result.append({"index":index,"task_id":task["task_id"],"partition":task["partition"],"state":state})
        except (ValueError,OSError,KeyError,TypeError) as exc:
            result.append({"index":index,"task_id":task["task_id"],"partition":task["partition"],"state":"invalid","error":str(exc)})
    return result


def make_schedule(plan_path,output,*,tasks_per_slot=4):
    if type(tasks_per_slot) is not int or tasks_per_slot<1:raise ValueError("tasks per slot must be positive")
    plan=read_plan(plan_path);states=status(plan)
    if any(s["state"]=="invalid" for s in states):raise ValueError("changed/invalid evidence must be investigated before rescheduling")
    remaining=[s["index"] for s in states if s["state"]!="complete"]
    slots=[remaining[i:i+tasks_per_slot] for i in range(0,len(remaining),tasks_per_slot)]
    schedule=_seal({"schema":SCHEDULE_SCHEMA,"plan":_identity(plan_path),"slots":slots,"maximum_concurrent_slots":4,
                    "array_argument":f"0-{len(slots)-1}%4" if slots else None})
    _write(output,schedule);return schedule


def run_task(plan,index):
    task=plan["tasks"][index];root=Path(plan["output"])/"tasks"/task["task_id"]
    with _task_lock(root/"producer.lock"):
        for binding in task["inputs"]:verify_identity(binding)
        done=_completed(plan,task)
        if done:return {"task_id":task["task_id"],"state":"already_complete","attempt":str(done[0])}
        numbers=[int(p.name.split("_")[-1]) for p in root.glob("attempt_*") if p.is_dir()]
        attempt=root/f"attempt_{max(numbers,default=0)+1:04d}";attempt.mkdir()
        execution=plan["execution"]
        command=[execution["python"],"-m","hoi.v1.native","--bundle",plan["bundles"][task["partition"]]["path"],
            "--asset-id",str(task["asset_id"]),"--num-envs","8","--steps","64","--seed",str(execution["seed"]),"--output",str(attempt/"native"),*execution["native_args"]]
        _write(attempt/"RUN.json",{"command":command,"plan_sha256":plan["content_sha256"],"task_sha256":task["task_sha256"],
            "slurm_job_id":os.environ.get("SLURM_JOB_ID"),"slurm_array_task_id":os.environ.get("SLURM_ARRAY_TASK_ID")})
        try:
            with (attempt/"native.log").open("xb") as log:
                result=subprocess.run(command,cwd=execution["source"],stdout=log,stderr=subprocess.STDOUT,check=False)
            if result.returncode:raise RuntimeError(f"native child exited {result.returncode}; see {attempt/'native.log'}")
            report=_json(attempt/"native/native.json");gate=_validate_report(plan,task,report)
            _write(attempt/"gate.json",gate)
            artifacts={"native":_identity(attempt/"native/native.json"),"gate":_identity(attempt/"gate.json"),
                "source":_identity(attempt/"native/source.json"),"geometry":_identity(report["compiled_geometry_export"]),
                "input_contract":verify_identity(report["native_input_contract"]["artifact"]),**_visual_artifacts(plan,report),**_scene_reset_artifacts(plan,task,report)}
            receipt=_seal({"schema":SUCCESS_SCHEMA,"plan_sha256":plan["content_sha256"],"task_sha256":task["task_sha256"],"artifacts":artifacts,"training_started":False})
            _write(attempt/"SUCCESS.json",receipt)
            return {"task_id":task["task_id"],"state":"complete","attempt":str(attempt)}
        except Exception as exc:
            _write(attempt/"FAILURE.json",{"task_id":task["task_id"],"error":str(exc),"training_started":False})
            return {"task_id":task["task_id"],"state":"failed","attempt":str(attempt),"error":str(exc)}


def run_slot(schedule_path,slot):
    schedule=_read_sealed(schedule_path,SCHEDULE_SCHEMA);verify_identity(schedule["plan"])
    if schedule.get("maximum_concurrent_slots")!=4:raise ValueError("schedule concurrency contract changed")
    plan=read_plan(schedule["plan"]["path"]);flat=[i for values in schedule["slots"] for i in values]
    if len(flat)!=len(set(flat)) or any(type(i) is not int or not 0<=i<len(plan["tasks"]) for i in flat):raise ValueError("duplicate/out of range scheduled tasks")
    if type(slot) is not int or not 0<=slot<len(schedule["slots"]):raise ValueError("slot outside schedule")
    results=[]
    for index in schedule["slots"][slot]:
        # Recheck executable identity between independent processes as well.
        plan=read_plan(schedule["plan"]["path"])
        try:results.append(run_task(plan,index))
        except (OSError,ValueError,RuntimeError,KeyError) as exc:results.append({"task_id":plan["tasks"][index]["task_id"],"state":"failed","error":str(exc)})
        print(json.dumps(results[-1]),flush=True)
    return results


@_planning_identity_cache()
def collect(plan_path,output):
    from .gates import main as aggregate
    plan=read_plan(plan_path);states=status(plan)
    incomplete=[s for s in states if s["state"]!="complete"]
    if incomplete:raise ValueError("full native coverage incomplete: "+json.dumps(incomplete))
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);partitions={}
    for partition in PARTITIONS:
        reports=[];receipts=[]
        for task in plan["tasks"]:
            if task["partition"]!=partition:continue
            attempt,receipt=_completed(plan,task);reports.extend(("--native-report",receipt["artifacts"]["native"]["path"]))
            receipts.append(_identity(attempt/"SUCCESS.json"))
        destination=output/partition
        rc=aggregate(["--bundle",plan["bundles"][partition]["path"],*reports,"--output",str(destination)])
        if rc:raise ValueError("partition aggregate native gate failed: "+partition)
        partitions[partition]={"bundle":plan["bundles"][partition],"gate":_identity(destination/"gate.json"),
            "native_evidence":_identity(destination/"native_evidence.json"),"success_receipts":receipts,"asset_count":len(receipts)}
    summary=_seal({"schema":"hoi_full_native_acceptance_summary_v1","plan":_identity(plan_path),"partitions":partitions,"passed":True,"training_started":False})
    _write(output/"summary.json",summary);return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("plan");p.add_argument("--bundle",action="append",required=True);p.add_argument("--source",required=True);p.add_argument("--python",required=True)
    p.add_argument("--native-args-json",required=True);p.add_argument("--output",required=True);p.add_argument("--seed",type=int,default=20261031)
    for name in ("status","schedule","collect"):
        p=sub.add_parser(name);p.add_argument("--plan",required=True)
        if name!="status":p.add_argument("--output",required=True)
        if name=="schedule":p.add_argument("--tasks-per-slot",type=int,default=4)
    p=sub.add_parser("run-slot");p.add_argument("--schedule",required=True);p.add_argument("--slot",required=True,type=int)
    args=parser.parse_args(argv)
    try:
        if args.command=="plan":
            value=make_plan(args.bundle,args.source,args.python,_json(args.native_args_json),args.output,seed=args.seed)
            print(json.dumps({"plan":str(Path(args.output)/"plan.json"),"task_counts":value["task_counts"]}));return 0
        if args.command=="status":value=status(read_plan(args.plan))
        elif args.command=="schedule":value=make_schedule(args.plan,args.output,tasks_per_slot=args.tasks_per_slot)
        elif args.command=="collect":value=collect(args.plan,args.output)
        else:
            value=run_slot(args.schedule,args.slot)
            return int(any(v["state"]=="failed" for v in value))
        print(json.dumps(value,indent=2));return 0
    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
        print("Native batch failed: "+str(exc),file=sys.stderr);return 2


if __name__=="__main__":raise SystemExit(main())
