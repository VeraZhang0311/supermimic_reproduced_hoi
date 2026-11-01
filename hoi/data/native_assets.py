"""Source-bound native object audit and explicit physics/cooking recipes.

Label surfaces, physical assumptions and cooked native collision unions keep
separate identities. No missing mass, material or inertia is silently filled.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path

AUDIT_SCHEMA = 'hoi_curated_native_asset_audit_v1'
BUILD_SCHEMA = 'hoi_curated_native_object_build_v1'
PHYSICS_POLICY_SCHEMA = 'hoi_curated_native_physics_policy_20261031_v2'
HOI_RETARGET_MASS_KG = {'clothesstand':2., 'floorlamp':2.5, 'largebox':2., 'largetable':6.,
                      'monitor':3., 'plasticbox':1., 'smallbox':.5, 'smalltable':4., 'suitcase':3.,
                      'trashcan':1., 'tripod':1.5, 'whitechair':2.5, 'woodchair':4.}


def physical_policy():
    return {'schema':PHYSICS_POLICY_SCHEMA, 'authorization':'user_explicit_20261031',
            'hoi_retarget_mass_kg':dict(HOI_RETARGET_MASS_KG),
            'mass_policy':'preserve_authored_source_mass; HOI_Retarget_uses_user_table',
            'missing_inertia':'uniform_density_metric_collision_geometry; nonwatertight_source_uses_native_closed_convex_components',
            'compound_overlap':'sum_component_volumes; overlapping_material_counted_per_collider; not_Boolean_union',
            'missing_source_friction':{'object':.6, 'ground':.3},
            'hoi_retarget_source_lateral_friction':.9,
            'grail_source_material':{'object':1.,'ground':1.,'source':'modular_tracking_env_cfg.py terrain and simulation default',
                'sha256':'1feed14c6684fff20218e2af512a28f476c0af26749f69c6034ab66b85b78dd2',
                'lines':'462-465,480-483,1587'},
            'omnicontact_source_ground_friction':1.,
            'omnicontact_source_env_cfg_sha256':'d737b55175c0a539d0862698a527dbf00b09b95e3e111b175b45249591b69c6c',
            'omnicontact_source_object_spawner_sha256':'bd37f9c16bf7f31217a5e9181f6fb33af1340e2d39a5aebfc6cb99b9759574b8',
            'omnicontact_fixed_dynamic_friction':{'push_or_slide_box':.2,'soccer_all_tasks':.55,'carry_box':.45},
            'omnicontact_source_dynamic_ranges':{'push_or_slide_box':[.1,.3],'soccer':[.4,.7],'carry_box':[.3,.6]},
            'omnicontact_source_static_ranges':{'push_or_slide_box':[.2,.4],'soccer':[.5,.8],'carry_box':[.5,.8]},
            'omnicontact_source_mass_randomization':{'range':[.5,1.5],'operation':'scale','copied':False},
            'simulator_material_semantics':'coefficients_explicit; MuJoCo_and_PhysX_combine_equivalence_not_claimed',
            'omnicontact_mapping':'user_approved_fixed_midpoint; source_randomization_schedule_not_copied',
            'source_friction':'retain_explicit_source_value; user_approved_deterministic_mapping_for_Omni',
            'training_started':False}



def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def exact_surface(shapes):
    """Exact numerical topology, excluding names/provenance/cooker metadata.

    Sorting whole shapes permits reordering without broadening equivalence to
    approximately equal surfaces. Vertex/triangle ordering remains significant.
    """
    result = []
    allowed = {'mesh': ('vertices', 'faces'), 'cube': ('size',), 'sphere': ('radius',),
               'capsule': ('radius', 'height', 'axis'), 'cylinder': ('radius', 'height', 'axis')}
    for shape in shapes:
        kind = shape['kind']
        if kind not in allowed:
            raise ValueError('unsupported source surface ' + kind)
        item = {'kind': kind, **{k: shape[k] for k in allowed[kind] if k in shape}}
        for key in ('position', 'quaternion_wxyz', 'transform'):
            if key in shape:
                item[key] = shape[key]
        result.append(item)
    return sorted(result, key=digest)


def source_usd_physics(path, expected_sha=None):
    """Read authored values only; USD/PhysX fallback values are not evidence."""
    from pxr import Usd, UsdGeom, UsdPhysics
    path = Path(path)
    actual = sha(path)
    if expected_sha is not None and actual != expected_sha:
        raise ValueError('source USD SHA256 changed')
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError('unable to open source USD')
    def value(v):
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        if hasattr(v, 'GetReal'):
            return [float(v.GetReal()), *map(float, v.GetImaginary())]
        return [value(x) for x in v]
    result = []
    for prim in stage.TraverseAll():
        attrs = {a.GetName(): value(a.Get()) for a in prim.GetAttributes()
                 if a.HasAuthoredValueOpinion() and any(k in a.GetName().lower()
                     for k in ('mass', 'inertia', 'centerofmass', 'principalaxes', 'density', 'friction',
                               'restitution', 'kinematic', 'approximation'))}
        if attrs:
            result.append({'prim': str(prim.GetPath()), 'attributes': attrs})
    return {'path': str(path), 'sha256': actual, 'meters_per_unit': UsdGeom.GetStageMetersPerUnit(stage),
            'root_path': str(stage.GetDefaultPrim().GetPath()), 'authored': result}


def audit_geometry(record, *, inspect_usd=False):
    path = Path(record['object_geometry_path'])
    geometry = json.loads(path.read_bytes())
    if digest(geometry) != record['object_geometry_sha256']:
        raise ValueError('source geometry SHA256 mismatch: ' + str(path))
    shapes = exact_surface(geometry['shapes'])
    source = geometry.get('source', {})
    dependencies = geometry.get('dependencies', [])
    # These descriptors contain source paths/hashes and scale contracts, not
    # reference trajectories, contact events or fabricated dynamic parameters.
    result = {k: record[k] for k in ('dataset', 'object_geometry_path', 'object_geometry_sha256')}
    result.update({'exact_surface_sha256': digest(shapes), 'shape_count': len(shapes),
                   'shape_kinds': dict(Counter(s['kind'] for s in shapes)),
                   'vertices': sum(len(s.get('vertices', [])) for s in shapes),
                   'triangles': sum(len(s.get('faces', [])) for s in shapes),
                   'basis': geometry.get('basis'), 'source_scale': geometry.get('scale'),
                   'dependencies': dependencies,
                   'source': {k: source[k] for k in ('path', 'sha256', 'asset_id', 'urdf_path', 'urdf_sha256',
                       'source_scale', 'source_aug_scale', 'kind', 'binding_status', 'declared_size_m',
                       'official_asset_key', 'author_asset_key', 'asset_key', 'author_usd_path', 'author_usd_sha256') if k in source},
                   'physical_profile_status': 'unresolved', 'native_physics_verified': False})
    if record['dataset'] == 'hoi_retarget':
        import xml.etree.ElementTree as ET
        urdf = Path(source['urdf_path'])
        if sha(urdf) != source['urdf_sha256']:
            raise ValueError('source URDF changed')
        tree = ET.parse(urdf)
        result['source_urdf_physics'] = {
            'inertial_xml': [ET.tostring(e, encoding='unicode') for e in tree.findall('.//inertial')],
            'contact_xml': [ET.tostring(e, encoding='unicode') for e in tree.findall('.//contact')]}
    if record['dataset'] == 'ultra':
        recipe_path = Path(source['path'])
        if sha(recipe_path) != source['sha256']:
            raise ValueError('ULTRA frozen native recipe changed')
        recipe = json.loads(recipe_path.read_bytes())
        matches = [a for a in recipe['assets'] if digest(exact_surface([dict(s, kind='mesh') for s in a['collision_meshes']])) == result['exact_surface_sha256']]
        if len(matches) != 1:
            raise ValueError('ULTRA label/native frozen surface binding is ambiguous')
        asset = matches[0]
        if sha(asset['native_asset']) != asset['native_asset_sha256']:
            raise ValueError('ULTRA native asset changed')
        result.update({'physical_profile_status': 'existing_exact_surface_bound_native_recipe',
                       'physical_profile': asset['physical_profile'], 'object_asset': asset['object_asset'],
                       'native_asset': asset['native_asset'], 'native_asset_sha256': asset['native_asset_sha256']})
    if inspect_usd and record['dataset'] == 'grail_sitting':
        usd = [d for d in dependencies if str(d['path']).lower().endswith(('.usd', '.usda', '.usdc'))]
        if len(usd) != 1:
            raise ValueError('GRAIL USD dependency needs explicit composition audit')
        result['source_usd_physics'] = source_usd_physics(usd[0]['path'], usd[0]['sha256'])
    return result


def audit_registry(index_path, output, *, workers=4, inspect_usd=False):
    index = json.loads(Path(index_path).read_bytes())
    records = index['records']
    identities = [r['object_geometry_sha256'] for r in records]
    if not records or len(identities) != len(set(identities)):
        raise ValueError('geometry audit index must contain unique identities')
    if not 1 <= workers <= 8:
        raise ValueError('bounded reader count must be 1..8')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(lambda r: audit_geometry(r, inspect_usd=inspect_usd), records):
            results.append(result)
            if len(results) % 100 == 0:
                print(json.dumps({'completed': len(results), 'total': len(records)}), flush=True)
    groups = defaultdict(list)
    for result in results:
        groups[result['exact_surface_sha256']].append(result['object_geometry_sha256'])
    summary = {'schema': AUDIT_SCHEMA, 'index': {'path': str(Path(index_path).resolve()), 'sha256': sha(index_path)},
               'identities': len(results), 'by_dataset': dict(Counter(r['dataset'] for r in results)),
               'exact_surface_groups': len(groups),
               'surface_groups_by_dataset': {d: len({r['exact_surface_sha256'] for r in results if r['dataset'] == d}) for d in sorted({r['dataset'] for r in results})},
               'physical_status': dict(Counter(r['physical_profile_status'] for r in results)),
               'surface_group_members': dict(groups), 'records': results,
               'limitations': ['Exact surface grouping alone does not establish physical or scene equivalence.',
                               'No source default mass, density, inertia or material is inferred.',
                               'Native validation must record measured compiled geometry and physical readback.']}
    (output / 'audit.json').write_bytes(canonical(summary) + b'\n')
    return summary


def physics_group_key(*, collision_meshes, physical_profile, material_contract, scene_contract, cooking_contract):
    """Group native runs only with complete exact physics and scene equality."""
    from hoi.v1.prepare import _profile
    if 'ground_friction' not in physical_profile or not material_contract or not cooking_contract:
        raise ValueError('complete explicit physical/material/cooking contracts required')
    profile = _profile(physical_profile)
    surface = exact_surface([dict(m, kind='mesh') for m in collision_meshes])
    return digest({'surface': surface, 'physical_profile': profile, 'material_contract': material_contract,
                   'scene_contract': scene_contract, 'cooking_contract': cooking_contract})


def uniform_density_profile(geometry, *, mass, friction, ground_friction):
    """Derive missing COM/inertia from closed metric collision solids.

    This is an explicit homogeneous-solid engineering assumption. Disconnected
    shapes use one density, with volume-weighted COM and the parallel-axis rule.
    Overlapping component volumes are counted per collider, matching the stated
    compound-collider model; this does not claim an exact Boolean-union density.
    """
    import numpy as np
    import trimesh
    from hoi.v1.prepare import _profile
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('explicit positive source/user object mass required')
    parts = []
    for shape in exact_surface(geometry['shapes']):
        if any(k in shape for k in ('position', 'quaternion_wxyz', 'transform')):
            raise ValueError('uniform density requires composed object-local shapes')
        if shape['kind'] == 'cube':
            size = np.asarray(shape['size'], dtype=np.float64)
            if size.shape != (3,) or not np.isfinite(size).all() or (size <= 0).any():
                raise ValueError('positive metric cuboid dimensions required')
            scale = float(size.max())
            normalized_volume = float(np.prod(size / scale))
            volume = float(np.prod(size)); center = np.zeros(3)
            inertia = np.diag(volume / 12. * (np.sum(size*size)-size*size))
        elif shape['kind'] == 'mesh':
            vertices = np.asarray(shape['vertices'], dtype=np.float64)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3 or not np.isfinite(vertices).all():
                raise ValueError('finite metric collision vertices required')
            low, high = vertices.min(0), vertices.max(0)
            origin, scale = low + (high - low) / 2., float((high - low).max())
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('collision solid volume/moments are degenerate')
            # Integrate near the component, at unit extent. Object-frame
            # integration subtracts O(volume * COM**2) terms to obtain a tiny
            # central inertia; an absolute m^3 cutoff also rejects valid small
            # solids. Translation and unit scaling change neither geometry nor
            # the homogeneous/component-overlap assumption.
            mesh = trimesh.Trimesh(vertices=(vertices-origin)/scale, faces=shape['faces'], process=False)
            if not mesh.is_watertight or not mesh.is_winding_consistent:
                raise ValueError('uniform density needs a closed consistently oriented collision solid')
            # The library computes COM even for zero volume; suppress only
            # that intermediate divide and reject the degenerate result below.
            with np.errstate(divide='ignore', invalid='ignore'):
                if mesh.volume < 0:
                    mesh.invert()
                props = mesh.mass_properties
            normalized_volume, local_center, local_inertia = (float(props['volume']), np.asarray(props['center_mass']), np.asarray(props['inertia'])) if isinstance(props,dict) else (float(props.volume),np.asarray(props.center_mass),np.asarray(props.inertia))
            volume = normalized_volume * scale**3
            center = origin + local_center * scale
            inertia = local_inertia * scale**5
        else:
            raise ValueError('uniform-density solid is not implemented: ' + shape['kind'])
        if (not np.isfinite(normalized_volume) or normalized_volume <= 1e-15 or
                not np.isfinite(volume) or volume <= 0 or not np.isfinite(center).all() or not np.isfinite(inertia).all()):
            raise ValueError('collision solid volume/moments are degenerate')
        parts.append((volume, center, inertia))
    if not parts:
        raise ValueError('collision solid is empty')
    volume = sum(v for v, _, _ in parts)
    center = sum(v*c for v,c,_ in parts)/volume
    inertia = sum(i + v*(np.dot(c-center,c-center)*np.eye(3)-np.outer(c-center,c-center)) for v,c,i in parts)
    profile = _profile({'mass':mass, 'com':center.tolist(), 'inertia':(inertia*(mass/volume)).tolist(),
                        'friction':friction, 'ground_friction':ground_friction})
    evidence = {'model':'uniform_density_closed_metric_collision_components_v1', 'density_kg_m3':mass/volume,
                'summed_component_volume_m3':volume, 'component_count':len(parts),
                'surface_sha256':digest(exact_surface(geometry['shapes'])),
                'overlap_policy':'component_volume_sum_not_boolean_union',
                'numerical_integration':{'method':'component_bbox_center_max_extent_normalization_v1',
                    'normalized_volume_lower_bound':1e-15, 'volume_scale_power':3, 'central_inertia_scale_power':5},
                'not_measured_source_inertia':True}
    return profile, evidence


def source_physical_inputs(dataset, geometry, *, task=None, source_mass=None):
    """Resolve only audited source fields or the user's versioned choices."""
    policy=physical_policy(); source=geometry.get('source',{})
    evidence={'policy':policy,'source_mass_kg':source_mass,'source_dataset':dataset}
    if dataset == 'hoi_retarget':
        name=source.get('asset_id','').removeprefix('omomo/')
        if name not in HOI_RETARGET_MASS_KG:
            raise ValueError('HOI object class has no user-approved mass')
        mass=HOI_RETARGET_MASS_KG[name]; friction=.9; ground_friction=.3
        evidence.update({'mass_origin':'user_explicit_class_table','source_asset_id':source['asset_id'],
                         'friction_origin':'source_URDF_contact_lateral_friction'})
    elif dataset == 'omnicontact':
        if task is None:
            raise ValueError('Omni material mapping requires the explicit source task')
        if task.startswith('soccer/'):
            mass=.5;friction=.55;group='soccer_all_tasks'
        elif task=='box/case1_carry':
            mass=2.;friction=.45;group='carry_box'
        elif task in ('box/case2_push','box/case3_slide'):
            mass=2.;friction=.2;group='push_or_slide_box'
        else:
            raise ValueError('unmapped explicit Omni task')
        ground_friction=1.
        evidence.update({'mass_origin':'author_object_spawner.py_revision_19cde38364a5b48e47348b270c895eabe2ba3ad9',
                         'source_task':task,'friction_mapping_group':group,'friction_origin':'user_approved_fixed_source_dynamic_range_midpoint'})
    elif dataset == 'grail_sitting':
        if source_mass is None:
            raise ValueError('GRAIL needs verified authored source mass')
        mass=source_mass;friction=1.;ground_friction=1.
        evidence.update({'mass_origin':'authored_source_USD','friction_origin':'source_terrain_and_simulation_default_material'})
    else:
        raise ValueError('use the already bound ULTRA native physical profile')
    if not isinstance(mass,(int,float)) or mass <= 0:
        raise ValueError('source/user mass must be positive')
    return {'mass':float(mass),'friction':friction,'ground_friction':ground_friction},evidence


def write_cooking_fixture(source_record, build_path, robot_recipe_path, output):
    """Bind a single real reference to an isolated import/export-only fixture."""
    build_path, robot_recipe_path = Path(build_path), Path(robot_recipe_path)
    build = json.loads(build_path.read_bytes()); robot = json.loads(robot_recipe_path.read_bytes())
    if build.get('schema') != BUILD_SCHEMA or build.get('status') != 'native_cooking_and_geometry_validation_required':
        raise ValueError('cooking fixture requires an unverified source build receipt')
    if build['object_geometry_sha256'] != source_record['object_geometry_sha256']:
        raise ValueError('cooking fixture label/source geometry identity mismatch')
    if sha(build['native_asset']) != build['native_asset_sha256']:
        raise ValueError('native cooking source changed')
    asset = {'asset_id':0, 'object_asset':source_record['dataset']+':'+source_record['object_geometry_sha256'],
             'exact_scale':[1.,1.,1.], 'mesh_to_object':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
             'native_asset':build['native_asset'], 'native_asset_sha256':build['native_asset_sha256'],
             'physical_profile':build['physical_profile'],
             'object_geometry_sha256':build['object_geometry_sha256'],
             'collision_hash':digest({'source_surface':build['exact_surface_sha256'],'cooking':build['collision_approximation']}),
             'build_receipt':{'path':str(build_path.resolve()),'sha256':sha(build_path)}}
    if build.get('inertia_mode') is not None:
        asset['inertia_mode']=build['inertia_mode']
    result={'schema':'hoi_curated_native_cooking_fixture_v1', 'source_record':deepcopy(source_record), 'asset':asset,
            'native_robot_asset':deepcopy(robot['native_robot_asset']),
            'reference_kinematics':deepcopy(robot['reference_kinematics']),
            'robot_geometry_sha256':robot['robot_geometry_sha256'],
            'robot_recipe':{'path':str(robot_recipe_path.resolve()),'sha256':sha(robot_recipe_path)},
            'scope':'curated_native_cooking_fixture_not_training'}
    result['content_sha256']=digest(result)
    output=Path(output)
    with output.open('xb') as stream:
        stream.write(canonical(result)+b'\n')
    return result


def bind_compiled_asset(build_path, compiled_recipe_path, *, final_output=None, repair_binding=None):
    """Promote actual exported convex hulls to a SHA-bound recipe for a later gate.

    Promotion supplies the exact compiled geometry cache. It is not a claim that
    the new reimport, label surface equivalence or training gate has passed.
    """
    build_path, compiled_recipe_path = Path(build_path), Path(compiled_recipe_path)
    build=json.loads(build_path.read_bytes()); compiled=json.loads(compiled_recipe_path.read_bytes())
    if build.get('schema')!=BUILD_SCHEMA or compiled.get('schema')!='hoi_v1_compiled_geometry_recipe_v1' or len(compiled.get('assets',[]))!=1:
        raise ValueError('unexpected source-build/native-export schema')
    entry=deepcopy(compiled['assets'][0])
    original_build=build; original_build_path=build_path
    if repair_binding is not None:
        from hoi.data.native_repairs import read_binding, _verify
        _verify(repair_binding)
        binding,proposal,_,original_build,_,actual_build,_=read_binding(repair_binding['path'])
        if binding['cooking_input_build'] != {'path':str(build_path.resolve()),'sha256':sha(build_path)} or actual_build != build:
            raise ValueError('repair finalization input build differs from approved chain')
        if (entry.get('repair_binding')!=repair_binding or entry.get('cooking_input_geometry_sha256')!=build['object_geometry_sha256'] or
            entry.get('cooking_input_surface_sha256')!=build['exact_surface_sha256'] or entry.get('source_exact_surface_sha256')!=original_build['exact_surface_sha256']):
            raise ValueError('actual repaired native export lost its dual source/input identity')
        from hoi.data.native_repairs import compare_candidate_export
        entry['repair_import_comparison']=compare_candidate_export(repair_binding['path'],entry['collision_meshes'])
        original_build_path=Path(proposal['original_build']['path'])
    elif build.get('physical_evidence',{}).get('collision_repair_proposal') or entry.get('repair_binding'):
        raise ValueError('repair asset finalization requires its approved repair binding')
    if entry['source_native_asset_sha256']!=build['native_asset_sha256'] or sha(entry['source_native_asset'])!=build['native_asset_sha256']:
        raise ValueError('compiled native source identity mismatch')
    mode=build.get('inertia_mode','explicit')
    if mode == 'explicit':
        if digest(entry['physical_profile'])!=digest(build['physical_profile']):
            raise ValueError('compiled physical profile changed')
    elif mode == 'native_cooking_only_uniform_density':
        if final_output is None:
            raise ValueError('deferred source inertia requires a distinct final native asset output')
        if any(entry['physical_profile'].get(k)!=v for k,v in build['physical_profile'].items()):
            raise ValueError('source/user mass or material changed during cooking')
    else:
        raise ValueError('unknown source build inertia mode')
    if sha(entry['native_asset'])!=entry['native_asset_sha256'] or not entry.get('native_convex_evidence'):
        raise ValueError('compiled collision export identity/evidence missing')
    entry['object_geometry_sha256']=original_build['object_geometry_sha256']
    entry['physical_evidence']=deepcopy(build['physical_evidence'])
    if mode == 'native_cooking_only_uniform_density' or repair_binding is not None:
        if final_output is None:raise ValueError('approved repair requires a distinct final native asset output')
        cooked={'shapes':[dict(m,kind='mesh') for m in entry['collision_meshes']]}
        if mode == 'native_cooking_only_uniform_density':
            profile,derived=uniform_density_profile(cooked,**build['physical_profile'])
            evidence={**entry['physical_evidence'], 'inertia_derivation':derived, 'policy':physical_policy()}
        else:
            profile=deepcopy(build['physical_profile'])
            evidence={**entry['physical_evidence'], 'inertia_policy':'preserve_explicit_source_profile_unchanged'}
        final=write_native_source(cooked,final_output,physical_profile=profile,physical_evidence=evidence,collision_approximation='convexHull',preserve_hull_vertices=repair_binding is not None)
        entry['native_asset'],entry['native_asset_sha256']=final['native_asset'],final['native_asset_sha256']
        entry['physical_profile'],entry['physical_evidence']=profile,evidence
        entry['final_native_build_receipt']={'path':str(Path(final_output).resolve()/'build.json'),'sha256':sha(Path(final_output)/'build.json')}
        entry.pop('inertia_mode',None)
    entry['source_build_receipt']={'path':str(original_build_path.resolve()),'sha256':sha(original_build_path)}
    if repair_binding is not None:
        entry['repair_binding']=deepcopy(repair_binding)
        entry['cooking_input_build_receipt']={'path':str(build_path.resolve()),'sha256':sha(build_path)}
        entry['hull_vertex_policy']='per_component_input_vertex_count_no_decimation'
    entry['compiled_recipe_receipt']={'path':str(compiled_recipe_path.resolve()),'sha256':sha(compiled_recipe_path)}
    entry['label_surface_sha256']=original_build['exact_surface_sha256']
    entry['native_validation_status']='reimport_geometry_and_pair_gate_required'
    return entry


def write_native_source(geometry, output, *, physical_profile, physical_evidence, collision_approximation, inertia_mode='explicit', preserve_hull_vertices=False):
    """Author an isolated native cooking input; never label it a verified bundle.

    A returned convex-decomposition input must be imported and exported through
    the actual native runtime before its compiled hulls become the cache recipe.
    """
    from hoi.v1.prepare import _profile
    if type(preserve_hull_vertices) is not bool:
        raise ValueError('explicit hull preservation switch must be boolean')
    if preserve_hull_vertices:
        from hoi.data.native_repairs import require_approval
        if collision_approximation != 'convexHull':raise ValueError('repair hull preservation requires explicit convex hull components')
        require_approval(physical_evidence['collision_repair_proposal']['path'],physical_evidence['collision_repair_approval']['path'])
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    import numpy as np
    from scipy.spatial.transform import Rotation
    if not physical_evidence or 'ground_friction' not in physical_profile:
        raise ValueError('explicit physical profile and evidence/assumption receipt required')
    if inertia_mode == 'explicit':
        profile = _profile(physical_profile)
    elif inertia_mode == 'native_cooking_only_uniform_density':
        if set(physical_profile) != {'mass','friction','ground_friction'} or not np.isfinite(list(physical_profile.values())).all() or physical_profile['mass'] <= 0 or min(physical_profile['friction'],physical_profile['ground_friction']) < 0:
            raise ValueError('cooking-only mass/material contract must be explicit; no guessed COM/inertia')
        profile = deepcopy(physical_profile)
    else:
        raise ValueError('unknown inertia mode')
    if collision_approximation not in ('convexHull', 'convexDecomposition'):
        raise ValueError('native dynamic mesh needs an explicit convex cooking policy')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    stage = Usd.Stage.CreateNew(str(output / 'object.usda'))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, '/object')
    stage.SetDefaultPrim(root.GetPrim())
    rigid = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(profile['mass'])
    if inertia_mode == 'explicit':
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float,profile['com'])))
        inertia, axes = np.linalg.eigh(profile['inertia'])
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        q = Rotation.from_matrix(axes).as_quat()
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float,inertia)))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float,q[:3]))))
    for i, shape in enumerate(exact_surface(geometry['shapes'])):
        if any(k in shape for k in ('position', 'quaternion_wxyz', 'transform')):
            raise ValueError('native input requires already composed object-local shape coordinates')
        path = f'/object/collision_{i:04d}'
        approximation = collision_approximation
        if shape['kind'] == 'cube':
            # Native export verifies graph-backed hulls. This is the exact
            # eight-corner cuboid, not a mesh approximation of a curved solid.
            import trimesh
            box = trimesh.creation.box(extents=shape['size'])
            vertices, faces = box.vertices.tolist(), box.faces.tolist()
            approximation = 'convexHull'
        elif shape['kind'] == 'mesh':
            vertices, faces = shape['vertices'], shape['faces']
        else:
            raise ValueError('native builder needs an explicit implementation for ' + shape['kind'])
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in vertices])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([int(x) for face in faces for x in face])
        mesh.CreateSubdivisionSchemeAttr('none')
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approximation)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        if preserve_hull_vertices:
            from hoi.data.native_repairs import apply_hull_vertex_contract
            apply_hull_vertex_contract(mesh.GetPrim(),len(vertices))
    stage.GetRootLayer().Save()
    manifest = {'schema': BUILD_SCHEMA, 'object_geometry_sha256': digest(geometry),
                'exact_surface_sha256': digest(exact_surface(geometry['shapes'])),
                'native_asset': str((output/'object.usda').resolve()), 'native_asset_sha256': sha(output/'object.usda'),
                'physical_profile': profile, 'physical_evidence': deepcopy(physical_evidence),
                'collision_approximation': collision_approximation,
                'status': 'native_cooking_and_geometry_validation_required', 'training_ready': False}
    if preserve_hull_vertices:
        manifest['hull_vertex_policy']='per_component_input_vertex_count_no_decimation'
    if inertia_mode != 'explicit':
        manifest['inertia_mode']=inertia_mode
    (output/'build.json').write_bytes(canonical(manifest)+b'\n')
    return manifest


def prepare_native_sources(split_path, robot_recipe_path, output, *, clip_ids=None):
    """Build source USDs and cooking fixtures, keeping every label binding.

    Exact object geometry plus the complete resolved object physical contract
    can share one cooking input. This is object-import deduplication only: scene
    contracts and actual native acceptance remain separately gated.
    """
    from hoi.data.curated import read_split
    split, partitions=read_split(split_path)
    records=partitions['train']+partitions['protected_eval']
    if clip_ids is not None:
        selected=set(clip_ids);records=[r for r in records if r['clip_id'] in selected]
        if not selected or {r['clip_id'] for r in records} != selected:
            raise ValueError('native source selection contains unknown clip IDs')
    unique={}
    for record in records:
        unique.setdefault(record['object_geometry_sha256'],record)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    policy=physical_policy();(output/'physical_policy.json').write_bytes(canonical(policy)+b'\n')
    robot_recipe=json.loads(Path(robot_recipe_path).read_bytes())
    from hoi.data.curated_bundle import expand_native_recipe_assets
    ultra={a['object_geometry_sha256']:a for a in expand_native_recipe_assets(robot_recipe) if a.get('object_geometry_sha256')}
    groups={};bindings=[];usd_cache={}
    for index, (label_hash, record) in enumerate(sorted(unique.items())):
        geometry=json.loads(Path(record['object_geometry_path']).read_bytes())
        if digest(geometry)!=label_hash:
            raise ValueError('label geometry changed during native source build')
        if record['dataset']=='ultra':
            if label_hash not in ultra:
                raise ValueError('ULTRA needs its existing exact native binding')
            a=ultra[label_hash]
            bindings.append({'object_geometry_sha256':label_hash,'dataset':'ultra','status':'existing_native_recipe_requires_region16_gate',
                             'asset_recipe':a,'source_record':record})
            continue
        source_mass=None;source_evidence={}
        if record['dataset']=='grail_sitting':
            dependencies=geometry.get('dependencies',[])
            usd=[d for d in dependencies if str(d['path']).lower().endswith(('.usd','.usda','.usdc'))]
            if len(usd)!=1:
                raise ValueError('GRAIL source USD composition requires an explicit audit')
            key=usd[0]['sha256']
            if key not in usd_cache:
                usd_cache[key]=source_usd_physics(usd[0]['path'],key)
            source_evidence=usd_cache[key]
            root_attrs=next((r['attributes'] for r in source_evidence['authored'] if r['prim']==source_evidence['root_path']),{})
            source_mass=root_attrs.get('physics:mass')
        base,evidence=source_physical_inputs(record['dataset'],geometry,task=record.get('task'),source_mass=source_mass)
        if source_evidence:
            evidence['authored_source_usd']=source_evidence
        try:
            profile,derived=uniform_density_profile(geometry,**base)
            evidence['inertia_derivation']=derived;mode='explicit'
        except ValueError as error:
            if 'closed consistently oriented collision solid' not in str(error):
                raise
            profile=base;mode='native_cooking_only_uniform_density'
            evidence['inertia_derivation']={'status':'waiting_for_native_closed_convex_components','reason':str(error)}
        # Geometry is already metric/object-local; all scales were applied once
        # by the immutable source adapter and are never applied again here.
        shape_hash=digest(exact_surface(geometry['shapes']))
        approximation='convexHull' if geometry.get('collision_approximation')=='convexHull' or all(sh['kind']=='cube' for sh in geometry['shapes']) else 'convexDecomposition'
        group_key=digest({'exact_surface_sha256':shape_hash,'physical_profile':profile,'inertia_mode':mode,'collision_approximation':approximation})
        if group_key not in groups:
            folder=output/f'object_{len(groups):04d}'
            build=write_native_source(geometry,folder,physical_profile=profile,physical_evidence=evidence,
                                      collision_approximation=approximation,inertia_mode=mode)
            fixture=write_cooking_fixture(record,folder/'build.json',robot_recipe_path,folder/'cooking_fixture.json')
            groups[group_key]={'object_build_group_sha256':group_key,'representative_geometry_sha256':label_hash,
                               'build_receipt':{'path':str((folder/'build.json').resolve()),'sha256':sha(folder/'build.json')},
                               'cooking_fixture':{'path':str((folder/'cooking_fixture.json').resolve()),'sha256':sha(folder/'cooking_fixture.json')},
                               'inertia_mode':mode,'members':[]}
        groups[group_key]['members'].append(label_hash)
        bindings.append({'object_geometry_sha256':label_hash,'dataset':record['dataset'],'exact_surface_sha256':shape_hash,
                         'object_build_group_sha256':group_key,'source_record':record,'source_physical_evidence':evidence,
                         'status':'native_cooking_and_final_gate_required'})
        if (index+1)%50==0:
            print(json.dumps({'prepared_identities':index+1,'total':len(unique),'object_cooking_groups':len(groups)}),flush=True)
    result={'schema':'hoi_curated_native_source_registry_v1','split':{'path':str(Path(split_path).resolve()),'sha256':sha(split_path),'content_sha256':split['content_sha256']},
            'robot_recipe':{'path':str(Path(robot_recipe_path).resolve()),'sha256':sha(robot_recipe_path)},
            'policy':{'path':str((output/'physical_policy.json').resolve()),'sha256':sha(output/'physical_policy.json')},
            'selected_geometry_identities':len(unique),'full_selection':clip_ids is None,'object_cooking_groups':list(groups.values()),
            'bindings':bindings,'training_ready':False,'training_started':False,
            'scope':'source_asset_registry; object-only exact equality does not substitute for scene/native acceptance'}
    result['content_sha256']=digest(result)
    (output/'registry.json').write_bytes(canonical(result)+b'\n')
    return result


def finalize_native_registry(registry_path, exports, output):
    """Bind each label identity to measured cooked hulls and explicit final physics.

    ``exports`` maps object_build_group_sha256 to its real compiled assets.json.
    Missing groups are errors; a partial registry never pretends to cover all
    2448 identities. The final standard bundle gate is still required.
    """
    registry_path=Path(registry_path);registry=json.loads(registry_path.read_bytes())
    if registry.get('schema')!='hoi_curated_native_source_registry_v1' or registry.get('content_sha256')!=digest({k:v for k,v in registry.items() if k!='content_sha256'}):
        raise ValueError('native source registry identity changed')
    from hoi.data.native_repairs import validate_derived_registry
    validate_derived_registry(registry)
    groups={g['object_build_group_sha256']:g for g in registry['object_cooking_groups']}
    if set(exports)!=set(groups):
        raise ValueError('every declared native object cooking group needs exactly one actual export')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    prepared={}
    for index,(key,group) in enumerate(sorted(groups.items())):
        receipt=group['build_receipt']
        if sha(receipt['path'])!=receipt['sha256']:
            raise ValueError('source native build changed')
        entry=bind_compiled_asset(receipt['path'],exports[key],final_output=output/f'final_{index:04d}',repair_binding=group.get('repair_binding'))
        prepared[key]=entry
    robot=registry['robot_recipe']
    if sha(robot['path'])!=robot['sha256']:
        raise ValueError('native robot recipe changed')
    recipe=json.loads(Path(robot['path']).read_bytes());assets=[];shared=[];shared_lookup={};aliases=[]
    for binding in registry['bindings']:
        if binding['dataset']=='ultra':
            entry=dict(binding['asset_recipe']);sharing_key=('ultra',binding['object_geometry_sha256'])
        else:
            key=binding['object_build_group_sha256'];group=groups[key]
            if binding['object_geometry_sha256'] not in group['members']:
                raise ValueError('label identity lacks exact native object group membership')
            entry=dict(prepared[key]);sharing_key=('compiled_object',key)
            if binding['exact_surface_sha256']!=entry['label_surface_sha256']:
                raise ValueError('native object alias differs from exact source collision surface')
            entry['object_geometry_sha256']=binding['object_geometry_sha256']
            entry['object_asset']=binding['dataset']+':'+binding['object_geometry_sha256']
            entry['label_native_binding']={'object_build_group_sha256':key,
                'representative_geometry_sha256':group['representative_geometry_sha256'],
                'exact_source_surface_sha256':binding['exact_surface_sha256'],
                'proof':('identical_original_source_surface_and_shared_approved_repair; native_surface_may_differ; scene_acceptance_separate' if entry.get('repair_binding') else 'identical_metric_surface_and_resolved_object_physics; scene_acceptance_separate')}
            entry['source_physical_evidence']=binding['source_physical_evidence']
        if sharing_key not in shared_lookup:
            shared_lookup[sharing_key]=len(shared);shared.append(entry)
        index=shared_lookup[sharing_key]
        overrides={k:entry[k] for k in ('object_geometry_sha256','object_asset','source_physical_evidence','label_native_binding') if k in entry}
        expanded={**shared[index],**overrides}
        if digest(expanded)!=digest(entry):
            raise ValueError('native alias changes shared physical/collision fields')
        aliases.append({'shared_asset_index':index,'overrides':overrides,'expanded_recipe_entry_sha256':digest(expanded)})
        assets.append(entry)
    if len(assets)!=registry['selected_geometry_identities']:
        raise ValueError('final native registry coverage mismatch')
    from hoi.data.curated_bundle import NATIVE_ALIAS_SCHEMA, expand_native_recipe_assets
    recipe['assets']=shared
    recipe['native_alias_schema']=NATIVE_ALIAS_SCHEMA
    recipe['object_geometry_aliases']=aliases
    if len(expand_native_recipe_assets(recipe))!=registry['selected_geometry_identities']:
        raise ValueError('compact recipe lost a label identity')
    recipe['native_source_registry']={'path':str(registry_path.resolve()),'sha256':sha(registry_path),
                                      'content_sha256':registry['content_sha256']}
    recipe['native_validation_status']='all_final_assets_require_standard_geometry_pair_and_scene_gate'
    path=output/'recipes.json';path.write_bytes(canonical(recipe)+b'\n')
    return recipe


def main(argv=None):
    import sys
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='finalize':
        parser=argparse.ArgumentParser(description='Bind actual native exports to every declared label identity')
        parser.add_argument('--registry',type=Path,required=True)
        parser.add_argument('--exports',type=Path,required=True,help='JSON mapping object group SHA256 to compiled assets.json')
        parser.add_argument('--output',type=Path,required=True)
        args=parser.parse_args(argv[1:])
        recipe=finalize_native_registry(args.registry,json.loads(args.exports.read_bytes()),args.output)
        print(json.dumps({'recipe':str(args.output/'recipes.json'),'geometry_identities':len(recipe.get('object_geometry_aliases',recipe['assets'])),'shared_assets':len(recipe['assets'])}))
        return
    if argv and argv[0]=='build':
        parser=argparse.ArgumentParser(description='Prepare explicit source USDs and native cooking fixtures')
        parser.add_argument('--split',type=Path,required=True)
        parser.add_argument('--robot-recipe',type=Path,required=True)
        parser.add_argument('--output',type=Path,required=True)
        parser.add_argument('--clip-id',action='append')
        args=parser.parse_args(argv[1:])
        result=prepare_native_sources(args.split,args.robot_recipe,args.output,clip_ids=args.clip_id)
        print(json.dumps({'registry':str(args.output/'registry.json'),'identities':result['selected_geometry_identities'],'cooking_groups':len(result['object_cooking_groups'])}))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--inspect-usd', action='store_true')
    args = parser.parse_args(argv)
    result = audit_registry(args.index, args.output, workers=args.workers, inspect_usd=args.inspect_usd)
    print(json.dumps({k: result[k] for k in ('identities', 'by_dataset', 'exact_surface_groups', 'surface_groups_by_dataset', 'physical_status')}))


if __name__ == '__main__':
    main()
