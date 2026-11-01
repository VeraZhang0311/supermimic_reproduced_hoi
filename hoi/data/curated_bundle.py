"""Bind curated references to explicit native assets; never infer physical assets."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import torch

from .curated import (EXPECTED_SCHEMA, REFERENCE_SCHEMA, digest, load_json, load_curated_clip,
                      read_split, sha, canonical_bytes)
from .manifest import robot_joint_names
from hoi.v1.contacts import CHANNELS, LABEL_SCHEMA, BODY_GROUP_LINKS, GROUP_SCHEMA

CURATED_BUNDLE_SCHEMA = 'supermimic_hoi_curated_regions16_bundle_v2'


NATIVE_ALIAS_SCHEMA = 'hoi_curated_native_alias_recipes_v1'


def _canonical_alias_digest(shared, overrides, encoded_fields):
    """Hash the original canonical entry bytes, reusing unchanged field bytes.

    The cache belongs to one shared entry in one expansion call. It contains
    encoded values, never trusted subobject digests or cross-call state.
    """
    keys = shared.keys() | overrides.keys()
    # Parsed JSON objects have string keys. Preserve the existing encoder's
    # behavior for direct Python callers supplying other key types as well.
    if any(not isinstance(key, str) for key in keys):
        return digest({**shared, **overrides})
    result = hashlib.sha256()
    result.update(b'{')
    for number, key in enumerate(sorted(keys)):
        if number:
            result.update(b',')
        result.update(canonical_bytes(key))
        result.update(b':')
        if key in overrides:
            value = canonical_bytes(overrides[key])
        else:
            if key not in encoded_fields:
                encoded_fields[key] = canonical_bytes(shared[key])
            value = encoded_fields[key]
        result.update(value)
    result.update(b'}')
    return result.hexdigest()


def _expand_native_recipe_assets(recipe):
    """Return entries and call-local proofs for immutable shallow expansion."""
    if recipe.get('native_alias_schema') is None:
        from .native_repairs import validate_final_entry
        for asset in recipe['assets']:validate_final_entry(asset)
        return recipe['assets'], {}, {}
    if recipe['native_alias_schema'] != NATIVE_ALIAS_SCHEMA:
        raise ValueError('unsupported native recipe alias schema')
    shared=recipe['assets'];expanded=[];seen=set();used=set()
    from .native_repairs import validate_final_entry
    for asset in shared:validate_final_entry(asset)
    # These four fields are excluded from native_asset_equivalence. Only this
    # validated schema permits reuse of that identity by shared asset index.
    allowed={'object_geometry_sha256','object_asset','source_physical_evidence','label_native_binding'}
    encoded_fields={};entry_digests={};shared_indices={}
    for alias in recipe.get('object_geometry_aliases',[]):
        index=alias.get('shared_asset_index');overrides=alias.get('overrides')
        if type(index) is not int or not 0 <= index < len(shared) or not isinstance(overrides,dict) or not set(overrides) <= allowed or 'object_geometry_sha256' not in overrides:
            raise ValueError('native label alias may not override shared physics/geometry')
        label_hash=overrides['object_geometry_sha256']
        if not isinstance(label_hash,str) or len(label_hash)!=64 or label_hash in seen:
            raise ValueError('native recipe has duplicate/invalid label aliases')
        entry={**shared[index],**overrides}
        entry_digest=_canonical_alias_digest(shared[index],overrides,encoded_fields.setdefault(index,{}))
        if alias.get('expanded_recipe_entry_sha256') != entry_digest:
            raise ValueError('native alias expanded recipe hash changed')
        proof=entry.get('label_native_binding')
        if proof and proof.get('exact_source_surface_sha256') != entry.get('label_surface_sha256'):
            raise ValueError('native alias lacks exact source-surface equality')
        expanded.append(entry);seen.add(label_hash);used.add(index)
        entry_digests[label_hash]=entry_digest;shared_indices[label_hash]=index
    if used != set(range(len(shared))):
        raise ValueError('every shared native recipe needs a label identity')
    for i,asset in enumerate(shared):
        if not any(a['shared_asset_index']==i and a['overrides']['object_geometry_sha256']==asset['object_geometry_sha256'] for a in recipe['object_geometry_aliases']):
            raise ValueError('shared native representative label binding is missing')
    return expanded, entry_digests, shared_indices


def expand_native_recipe_assets(recipe):
    """Shallow-expand lightweight label aliases; shared physics is immutable."""
    return _expand_native_recipe_assets(recipe)[0]


def _cache_safe(value):
    """Retain source numeric precision in weights-only-loadable cache objects."""
    import numpy as np
    if isinstance(value,np.ndarray):
        return torch.as_tensor(value.copy()) if value.dtype.kind in 'biufc' else value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,dict):
        return {k:_cache_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [_cache_safe(v) for v in value]
    return value


def native_asset_equivalence(entry, root):
    """Conservative exact equality for native object compilation/worker identity."""
    from hoi.v1.prepare import _profile, _resolve
    scale = torch.as_tensor(entry['exact_scale'],dtype=torch.float64).reshape(-1)
    if scale.numel()==1:
        scale=scale.repeat(3)
    return digest({'native_asset':str(_resolve(entry['native_asset'],root).resolve()),
                   'native_asset_sha256':entry['native_asset_sha256'],
                   'physical_profile':_profile(entry['physical_profile']),
                   'exact_scale':scale.tolist(),
                   'mesh_to_object':torch.as_tensor(entry['mesh_to_object'],dtype=torch.float64).tolist(),
                   'collision_meshes':entry['collision_meshes'],
                   'cooking_recipe':entry.get('cooking_recipe','explicit_meshes_no_recook'),
                   'distance_recipe':entry.get('distance_recipe',{})})


def _prepare_asset(entry, asset_id, root, output, geometry_caches=None, *, require_cached_geometry=False):
    from hoi.v1.prepare import _profile, _resolve, read_triangle_mesh, _save
    from hoi.v1.geometry import build_geometry_cache
    if not isinstance(entry.get('object_geometry_sha256'), str) or len(entry['object_geometry_sha256']) != 64:
        raise ValueError('curated native asset must explicitly bind its label object geometry SHA256')
    native = _resolve(entry['native_asset'], root)
    if sha(native) != entry['native_asset_sha256']:
        raise ValueError('native asset SHA256 mismatch')
    if entry.get('inertia_mode') is not None:
        raise ValueError('cooking-only inertia modes cannot enter a curated reference bundle')
    for key in ('source_build_receipt','compiled_recipe_receipt','final_native_build_receipt'):
        if key in entry and sha(entry[key]['path']) != entry[key]['sha256']:
            raise ValueError('native asset provenance receipt changed: ' + key)
    profile = _profile(entry['physical_profile'])
    scale = torch.as_tensor(entry['exact_scale'], dtype=torch.float64)
    if scale.numel() == 1:
        scale = scale.repeat(3)
    matrix = torch.as_tensor(entry['mesh_to_object'], dtype=torch.float64)
    if scale.shape != (3,) or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()) or matrix.shape != (4,4):
        raise ValueError('exact metric scale and mesh-to-object transform required')
    rotation = matrix[:3,:3]
    if not bool(torch.isfinite(matrix).all()) or not torch.allclose(matrix[3], matrix.new_tensor([0,0,0,1])) or not torch.allclose(rotation.T@rotation, torch.eye(3,dtype=rotation.dtype),atol=1e-7,rtol=0) or float(torch.det(rotation)) < 0:
        raise ValueError('invalid object mesh transform')
    meshes, identities = [], []
    for record in entry['collision_meshes']:
        mesh, identity = read_triangle_mesh(record, root)
        frame = record.get('frame','mesh')
        if frame == 'mesh':
            mesh['vertices'] = (rotation@(mesh['vertices']*scale).T).T + matrix[:3,3]
        elif frame != 'object_local':
            raise ValueError('unknown collision mesh frame')
        meshes.append(mesh); identities.append({**identity,'frame':frame})
    collision_hash = digest({'mesh_identities':identities, 'exact_scale':scale.tolist(), 'mesh_to_object':matrix.tolist(),
                             'cooking_recipe':entry.get('cooking_recipe','explicit_meshes_no_recook')})
    if entry.get('collision_hash',collision_hash) != collision_hash:
        raise ValueError('native collision recipe hash mismatch')
    cache_key = digest({'collision_hash':collision_hash,'distance_recipe':entry.get('distance_recipe',{})})
    if geometry_caches is not None and cache_key in geometry_caches:
        cache = geometry_caches[cache_key]
        from hoi.v1.prepare import _bound_load
        from hoi.v1.geometry import tensor_sha256
        geometry = _bound_load(cache)
        metadata = geometry['metadata']
        if (metadata.get('collision_hash') != collision_hash or
                metadata.get('distance_recipe') != entry.get('distance_recipe', {}) or
                metadata.get('mesh_to_object') != matrix.tolist() or
                metadata.get('exact_scale') != scale.tolist() or metadata.get('mesh_identities') != identities or
                metadata.get('point_hash') != tensor_sha256(geometry['points'])):
            raise ValueError('prepared geometry cache differs from the exact collision/distance recipe')
        actual_meshes = geometry.get('collision_meshes', [])
        if len(actual_meshes) != len(meshes) or any(
                not torch.equal(a[key], b[key]) for a,b in zip(actual_meshes,meshes) for key in ('vertices','faces')):
            raise ValueError('prepared geometry cache collider tensors differ from the bound native asset')
    else:
        if require_cached_geometry:
            raise ValueError('prepared geometry manifest is missing the required collision cache')
        geometry = build_geometry_cache(meshes, **entry.get('distance_recipe',{}), metadata={'collision_hash':collision_hash,
                                         'mesh_to_object':matrix.tolist(),'exact_scale':scale.tolist(),'mesh_identities':identities,
                                         'distance_recipe':entry.get('distance_recipe', {})})
        cache = _save(output/f'geometry_{asset_id:04d}.pt',geometry)
        if geometry_caches is not None:
            geometry_caches[cache_key] = cache
    return {**entry,'asset_id':asset_id,'native_asset':str(native.resolve()),'physical_profile':profile,'collision_hash':collision_hash,
            'geometry_cache':cache}


def prepare_curated_bundle(split_path, recipes_path, output, kin, *, partition='train', path_maps=(), clip_ids=None,
                           geometry_cache_manifest=None):
    """Prepare the complete selected partition against supplied audited recipes.

    The recipes use the existing hoi.v1.prepare native asset/hand/FK contract,
    adding object_geometry_sha256 for an exact label-to-native binding. No USD
    mass, inertia, convex decomposition, scene support or scaling is guessed.
    """
    from hoi.v1.prepare import _hands, _save, _resolve, native_robot_asset_binding
    from .curated import _resolved
    from hoi.v1.scene_supports import digest as support_digest
    split, partitions = read_split(split_path)
    if partition not in partitions:
        raise ValueError('unknown curated partition')
    recipe = load_json(recipes_path); root = Path(recipes_path).resolve().parent
    if not isinstance(recipe.get('robot_geometry_sha256'), str) or len(recipe['robot_geometry_sha256']) != 64:
        raise ValueError('curated recipes must bind the offline robot geometry SHA256')
    available = partitions[partition]
    entries = available
    recipe_entries = expand_native_recipe_assets(recipe)
    if clip_ids is not None:
        requested = set(clip_ids)
        entries = [r for r in available if r['clip_id'] in requested]
        if not requested or {r['clip_id'] for r in entries} != requested:
            raise ValueError('native validation clip IDs must belong to the declared partition')
    required = {r['object_geometry_sha256'] for r in entries}
    supplied = [a['object_geometry_sha256'] for a in recipe_entries]
    all_required={r['object_geometry_sha256'] for values in partitions.values() for r in values}
    if len(set(supplied)) != len(supplied) or not required.issubset(supplied) or (clip_ids is None and set(supplied) != all_required):
        raise ValueError('every selected object geometry needs one unambiguous native asset recipe; complete bundles must cover the full curated selection')
    if 'reference_kinematics' not in recipe or 'native_robot_asset' not in recipe:
        raise ValueError('curated bundles require bound native robot and 39-body reference kinematics')
    virtual = deepcopy(recipe['reference_kinematics'])
    if len(virtual['body_names']) != 39 or len(set(virtual['body_names'])) != 39 or virtual['dof_names'] != robot_joint_names():
        raise ValueError('curated reference kinematics must retain canonical G1 body/joint identity')
    urdf = _resolve(virtual['urdf']['path'],root)
    if sha(urdf) != virtual['urdf']['sha256']:
        raise ValueError('reference URDF identity changed')
    virtual['urdf']['path'] = str(urdf.resolve())
    robot = recipe['native_robot_asset']
    robot_binding = native_robot_asset_binding(_resolve(robot['path'],root),robot['sha256'])
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    hands = _hands(recipe['hands'],root,recipe.get('profile')=='custom')
    hand_cache = _save(output/'hand_geometry.pt',hands)
    geometry_caches = {}
    if geometry_cache_manifest is not None:
        from .curated_geometry import load_geometry_manifest
        geometry_caches = load_geometry_manifest(geometry_cache_manifest, recipes_path)
    equivalent, assets, bindings, lookup = {}, [], [], {}
    selected_recipes = [a for a in recipe_entries if clip_ids is None or a['object_geometry_sha256'] in required]
    for entry in selected_recipes:
        key = native_asset_equivalence(entry,root)
        if key not in equivalent:
            asset_id = len(assets)
            asset = _prepare_asset(entry,asset_id,root,output,geometry_caches,
                                   require_cached_geometry=geometry_cache_manifest is not None)
            asset['native_equivalence_sha256'] = key
            asset['object_geometry_aliases'] = []
            assets.append(asset); equivalent[key] = asset_id
        asset_id = equivalent[key]
        label_hash = entry['object_geometry_sha256']
        assets[asset_id]['object_geometry_aliases'].append(label_hash)
        lookup[label_hash] = asset_id
        bindings.append({'object_geometry_sha256':label_hash,'asset_id':asset_id,
                         'recipe_entry_sha256':digest(entry),'native_equivalence_sha256':key})
    clips=[]
    for i, record in enumerate(entries):
        clip = load_curated_clip(record,path_maps)
        if clip['label_metadata']['robot_geometry_sha256'] != recipe['robot_geometry_sha256']:
            raise ValueError('label/native recipe robot geometry binding mismatch')
        reference=clip['reference']; reference['provenance']['manifest_content_sha256']=split['content_sha256']
        reference['source_scene_supports'] = _cache_safe(clip['scene_supports'])
        support_summary = {'present':bool(clip['scene_supports']),'count':len(clip['scene_supports']),
                           'source_sha256':support_digest(clip['scene_supports']),
                           'storage':'reference_cache.source_scene_supports'}
        clips.append({**record,'name':clip['name'],'source_motion':record['output'],'source_motion_sha256':record['labels_sha256'],
                      'asset_id':lookup[record['object_geometry_sha256']], 'state_frames':len(reference['robot']['root_pos']),
                      'resolved_source_output':str(_resolved(record['output'], path_maps)),
                      'robot_geometry_sha256':recipe['robot_geometry_sha256'],
                      'source_frames':reference['metadata']['source_frames'], 'scene_supports':support_summary,
                      'reference_cache':_save(output/f'reference_{i:05d}.pt',_cache_safe(reference)),
                      'contact_cache':_save(output/f'contact_{i:05d}.pt',clip['contact'])})
    bundle={'schema_version':CURATED_BUNDLE_SCHEMA,'partition':partition,'contact_channels':list(CHANNELS),
            'contact_label_schema':LABEL_SCHEMA,'expected_contact_schema':EXPECTED_SCHEMA,
            'contact_group_schema':GROUP_SCHEMA,'contact_group_links':[list(x) for x in BODY_GROUP_LINKS],
            'reference_joint_names':robot_joint_names(),'reference_kinematics':virtual,'native_robot_asset':robot_binding,
            'robot_geometry_sha256':recipe['robot_geometry_sha256'],
            'manifest':{'path':str(Path(split_path).resolve()),'sha256':sha(split_path),'content_sha256':split['content_sha256']},
            'recipes':{'path':str(Path(recipes_path).resolve()),'sha256':sha(recipes_path)},
            'assets':assets,'object_geometry_bindings':bindings,'hands':{'geometry_cache':hand_cache},'clips':clips,
            'timebase':{'control_hz':50.,'control_dt_s':.02,'source_hz':'per_clip_preserved','contact_alignment':'row_i_interval_(i-1,i]'},
            'scope':{'available_motions':len(available),'selected_motions':len(entries),'is_subset':len(entries)!=len(available),
                     'purpose':'native_validation' if clip_ids is not None else 'curated_reference_bundle'},
            'validation':{'cpu_data_geometry':'verified','native_physics':'not_run','pair_instrumentation':'not_run',
                          'canonical_hand_identity':recipe.get('profile')!='custom','training_started':False,
                          'scene_supports_present':any(c['scene_supports']['present'] for c in clips)}}
    if geometry_cache_manifest is not None:
        bundle['prepared_geometry'] = {'path':str(Path(geometry_cache_manifest).resolve()),'sha256':sha(geometry_cache_manifest)}
    if hasattr(kin,'mjcf_path'):
        bundle['robot_fk']={'mjcf_path':str(Path(kin.mjcf_path).resolve()),'mjcf_sha256':sha(kin.mjcf_path),
                            'tracked_bodies':[kin.body_names[i] for i in kin.tracked_idx]}
    bundle['bundle_content_sha256']=digest(bundle)
    (output/'bundle.json').write_bytes(canonical_bytes(bundle)+b'\n')
    return bundle


def read_curated_bundle_header(path):
    bundle=load_json(path)
    payload={k:v for k,v in bundle.items() if k!='bundle_content_sha256'}
    if bundle.get('schema_version')!=CURATED_BUNDLE_SCHEMA or bundle.get('bundle_content_sha256')!=digest(payload):
        raise ValueError('curated bundle schema/content hash mismatch')
    if bundle.get('contact_channels')!=list(CHANNELS) or bundle.get('contact_label_schema')!=LABEL_SCHEMA or bundle.get('expected_contact_schema')!=EXPECTED_SCHEMA:
        raise ValueError('curated bundle contact schema mismatch')
    if bundle.get('contact_group_schema') != GROUP_SCHEMA or bundle.get('contact_group_links') != [list(x) for x in BODY_GROUP_LINKS]:
        raise ValueError('curated region mapping mismatch')
    if sha(bundle['recipes']['path']) != bundle['recipes']['sha256']:
        raise ValueError('curated native recipes changed')
    if 'prepared_geometry' in bundle and sha(bundle['prepared_geometry']['path']) != bundle['prepared_geometry']['sha256']:
        raise ValueError('curated prepared geometry manifest changed')
    recipe = load_json(bundle['recipes']['path'])
    if recipe.get('robot_geometry_sha256') != bundle.get('robot_geometry_sha256'):
        raise ValueError('curated robot geometry recipe binding mismatch')
    assets = bundle.get('assets', [])
    if not assets or [a['asset_id'] for a in assets] != list(range(len(assets))):
        raise ValueError('curated assets must have contiguous unique IDs')
    if len({a['native_equivalence_sha256'] for a in assets}) != len(assets):
        raise ValueError('curated native assets must use unique exact equivalence groups')
    from hoi.v1.prepare import _profile, _resolve
    recipe_entries, entry_digests, shared_indices = _expand_native_recipe_assets(recipe)
    recipe_assets = {a['object_geometry_sha256']:a for a in recipe_entries}
    if len(recipe_assets) != len(recipe_entries):
        raise ValueError('native recipe geometry bindings are ambiguous')
    recipe_root = Path(bundle['recipes']['path']).resolve().parent
    for asset in assets:
        original = recipe_assets.get(asset['object_geometry_sha256'])
        if original is None:
            raise ValueError('bundle asset lacks its source native recipe')
        for key, value in original.items():
            if key == 'native_asset':
                value = str(_resolve(value,recipe_root).resolve())
            elif key == 'physical_profile':
                value = _profile(value)
            if asset.get(key) != value:
                raise ValueError('bundle asset differs from native recipe: ' + key)
    bindings = bundle.get('object_geometry_bindings', [])
    aliases = {}
    per_asset = {i:[] for i in range(len(assets))}
    equivalences = {}
    for binding in bindings:
        label_hash = binding.get('object_geometry_sha256')
        asset_id = binding.get('asset_id')
        original = recipe_assets.get(label_hash)
        if label_hash in aliases or original is None or type(asset_id) is not int or not 0 <= asset_id < len(assets):
            raise ValueError('invalid/duplicate native object geometry alias binding')
        if label_hash in shared_indices:
            shared_index = shared_indices[label_hash]
            if shared_index not in equivalences:
                equivalences[shared_index] = native_asset_equivalence(original,recipe_root)
            equivalence = equivalences[shared_index]
            entry_digest = entry_digests[label_hash]
        else:
            equivalence = native_asset_equivalence(original,recipe_root)
            entry_digest = digest(original)
        if binding.get('recipe_entry_sha256') != entry_digest or binding.get('native_equivalence_sha256') != equivalence or assets[asset_id].get('native_equivalence_sha256') != equivalence:
            raise ValueError('native alias physics/geometry differs from its exact asset group')
        aliases[label_hash] = asset_id
        per_asset[asset_id].append(label_hash)
    expected_aliases = set(recipe_assets) if bundle['scope'].get('purpose') != 'native_validation' else {c['object_geometry_sha256'] for c in bundle['clips']}
    if set(aliases) != expected_aliases:
        raise ValueError('curated native aliases must cover every declared recipe geometry identity')
    for asset in assets:
        if not per_asset[asset['asset_id']] or asset.get('object_geometry_aliases') != per_asset[asset['asset_id']] or asset['object_geometry_sha256'] not in per_asset[asset['asset_id']]:
            raise ValueError('native asset alias membership mismatch')
    if bundle.get('reference_joint_names')!=robot_joint_names():
        raise ValueError('curated joint order mismatch')
    manifest=bundle['manifest']
    if sha(manifest['path'])!=manifest['sha256']:
        raise ValueError('curated split bytes changed')
    split,parts=read_split(manifest['path'])
    if split['content_sha256']!=manifest['content_sha256'] or bundle['partition'] not in parts:
        raise ValueError('curated split/partition identity mismatch')
    if bundle['scope'].get('purpose') != 'native_validation' and set(aliases) != {r['object_geometry_sha256'] for values in parts.values() for r in values}:
        raise ValueError('native aliases must cover the full immutable curated selection')
    entries={(r['dataset'],r['clip_id']):r for r in parts[bundle['partition']]}
    seen=set()
    for clip in bundle['clips']:
        summary=clip.get('scene_supports')
        if not isinstance(summary,dict) or type(summary.get('count')) is not int or summary['count'] < 0 or summary.get('present') is not (summary['count'] > 0) or summary.get('storage') != 'reference_cache.source_scene_supports' or not isinstance(summary.get('source_sha256'),str) or len(summary['source_sha256']) != 64:
            raise ValueError('curated support cache summary/binding missing')
        key=clip['dataset'],clip['clip_id']
        if key in seen or key not in entries:
            raise ValueError('curated bundle duplicated/wrong-partition clip')
        if any(clip[k]!=entries[key][k] for k in ('source_group','labels_sha256','metadata_sha256','output','object_geometry_sha256')):
            raise ValueError('curated bundle source identity mismatch')
        seen.add(key)
        if clip.get('robot_geometry_sha256') != bundle.get('robot_geometry_sha256'):
            raise ValueError('curated clip robot geometry mismatch')
        if type(clip['asset_id']) is not int or not 0 <= clip['asset_id'] < len(assets):
            raise ValueError('curated asset ID outside registry')
        if type(clip['state_frames']) is not int or clip['state_frames'] < 2:
            raise ValueError('curated clip requires at least two states')
        if aliases.get(clip['object_geometry_sha256'])!=clip['asset_id']:
            raise ValueError('curated label/native geometry binding mismatch')
    scope = bundle['scope']
    if scope.get('available_motions') != len(entries) or scope.get('selected_motions') != len(seen) or scope.get('is_subset') is not (len(seen) != len(entries)):
        raise ValueError('curated bundle scope/count mismatch')
    if seen != set(entries) and scope.get('purpose') != 'native_validation':
        raise ValueError('curated bundle must cover its complete declared partition')
    return bundle


def load_curated_bundle(path, kin, device='cpu', require_train=True, asset_filter=None, verify_source=True, retain_source_reference=None, shared_shapes=None):
    """Load local motion data with optional process-local static registry reuse.

    Pass a prior returned shapes object only for the same bundle/device/FK.
    Its geometry, hands, physical metadata and derived tensors must remain
    read-only; query's lazy SDF cache may grow. No global cache is maintained.
    Every call retains header, source, reference and native identity checks,
    and rehashes all static cache files even when skipping deserialization.
    Asset filtering affects motion frames/bins only, never the full registry.
    """
    from hoi.v1.prepare import (_bound_load, native_robot_asset_binding,
                                _bind_shared_shapes, _validate_shared_shapes)
    from hoi.v1.geometry import ShapeAssetBank, tensor_sha256
    from hoi.v1.motion import TrainingMotionBank
    from .curated import _resolved
    bundle=read_curated_bundle_header(path)
    if require_train and (bundle['partition']!='train' or bundle['scope'].get('purpose')=='native_validation'):
        raise ValueError('only train partition may supply replay/normalizer/learner data')
    if sha(bundle['recipes']['path'])!=bundle['recipes']['sha256']:
        raise ValueError('curated native recipes changed')
    native=bundle['native_robot_asset']
    if native_robot_asset_binding(native['path'],native['sha256'])!=native:
        raise ValueError('native robot identity changed')
    urdf=bundle['reference_kinematics']['urdf']
    if sha(urdf['path'])!=urdf['sha256']:
        raise ValueError('curated reference URDF changed')
    if 'robot_fk' in bundle and (not hasattr(kin,'mjcf_path') or sha(kin.mjcf_path)!=bundle['robot_fk']['mjcf_sha256']):
        raise ValueError('curated robot/runtime FK mismatch')
    if shared_shapes is not None:
        _validate_shared_shapes(shared_shapes,bundle,kin,device)
    clips,global_frames,offset=[],[],0
    selected=None if asset_filter is None else set(int(x) for x in asset_filter)
    for entry in bundle['clips']:
        n=entry['state_frames']
        if selected is not None and entry['asset_id'] not in selected:
            offset+=n;continue
        if verify_source:
            root=Path(entry['resolved_source_output'])
            if sha(root/'metadata.json')!=entry['metadata_sha256'] or sha(root/'labels.npz')!=entry['labels_sha256']:
                raise ValueError('curated source labels changed')
        reference,contacts=_bound_load(entry['reference_cache']),_bound_load(entry['contact_cache'])
        from hoi.v1.scene_supports import digest as support_digest
        if 'source_scene_supports' not in reference:
            raise ValueError('curated raw scene-support cache is missing')
        raw_supports=reference['source_scene_supports'];summary=entry['scene_supports']
        if not isinstance(raw_supports,list) or len(raw_supports) != summary['count'] or support_digest(raw_supports) != summary['source_sha256']:
            raise ValueError('curated scene-support cache differs from source binding')
        if reference['metadata']['schema_version']!=REFERENCE_SCHEMA or len(reference['robot']['root_pos'])!=n or reference['provenance']['manifest_content_sha256']!=bundle['manifest']['content_sha256'] or reference['provenance']['source_motion_sha256']!=entry['labels_sha256']:
            raise ValueError('curated reference cache identity mismatch')
        if contacts['metadata']['schema']!=EXPECTED_SCHEMA or contacts['metadata']['channels']!=list(CHANNELS):
            raise ValueError('curated contact cache schema mismatch')
        if contacts['expected_valid'][0].any():
            raise ValueError('first reference state cannot describe a preceding interval')
        clips.append({**entry,'reference':reference,'scene_supports':raw_supports,'expected_contact':contacts['expected_contact'],'expected_valid':contacts['expected_valid']})
        global_frames.append(torch.arange(offset,offset+n,dtype=torch.long,device=device));offset+=n
    bank=TrainingMotionBank(clips,kin,device,bundle['partition'],retain_source_reference=retain_source_reference)
    bank.global_frame_index=torch.cat(global_frames)
    bank.scene_supports=[clip.get('scene_supports', []) for clip in clips]
    bank.bound_scene_supports=[clip['reference'].get('scene_supports', []) for clip in clips]
    from hoi.v1.scene_supports import validate_native_scene_supports
    validate_native_scene_supports(bank)
    geometries=[]
    loaded_geometry, verified_native = {}, set()
    for asset in bundle['assets']:
        native_key=(asset['native_asset'],asset['native_asset_sha256'])
        if native_key not in verified_native:
            if sha(asset['native_asset'])!=asset['native_asset_sha256']:
                raise ValueError('native object asset changed')
            verified_native.add(native_key)
        cache_key=(asset['geometry_cache']['path'],asset['geometry_cache']['sha256'])
        if cache_key not in loaded_geometry:
            if shared_shapes is not None:
                if sha(cache_key[0]) != cache_key[1]:
                    raise ValueError(f'cache SHA256 mismatch: {cache_key[0]}')
                loaded_geometry[cache_key]=None
            else:
                loaded_geometry[cache_key]=_bound_load(asset['geometry_cache'])
        if shared_shapes is not None:
            continue  # Same bound file bytes and unchanged, already validated tensors.
        geometry=loaded_geometry[cache_key]
        if geometry['metadata']['collision_hash']!=asset['collision_hash'] or tensor_sha256(geometry['points'])!=geometry['metadata']['point_hash']:
            raise ValueError('curated geometry cache mismatch')
        geometries.append(geometry)
    if shared_shapes is not None:
        hand_cache=bundle['hands']['geometry_cache']
        if sha(hand_cache['path']) != hand_cache['sha256']:
            raise ValueError(f"cache SHA256 mismatch: {hand_cache['path']}")
        shapes=shared_shapes
    else:
        shapes=ShapeAssetBank(bundle['assets'],geometries,_bound_load(bundle['hands']['geometry_cache']),device)
        _bind_shared_shapes(shapes,bundle,kin,device)
    return bank,shapes,bundle
