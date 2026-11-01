"""Exact canonical-byte parity and fail-closed call-local header caching."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from hoi.data import curated_bundle as cb
from hoi.data.curated import canonical_bytes, digest, sha, write_split
from tests import test_hoi_curated_data as fixtures


def compact(shared, overrides):
    return {'native_alias_schema':cb.NATIVE_ALIAS_SCHEMA, 'assets':shared,
            'object_geometry_aliases':[{'shared_asset_index':index,'overrides':value,
                'expanded_recipe_entry_sha256':digest({**shared[index],**value})}
                for index,value in overrides]}


class CanonicalAliasTests(unittest.TestCase):
    def test_streamed_bytes_equal_original_encoder_for_nested_unicode_and_floats(self):
        shared={'object_geometry_sha256':'a'*64, '中文🔬':{'nested':[None,True,False,-0.0,1.e-100,1.e100,17],
                'quote\"\\':{'z':'line\nline','a':'é'}}, 'collision_meshes':[{'vertices':[[.123456789123,0.,-1.]],'faces':[]}],
                'object_asset':'old', 'source_physical_evidence':{'source':'原始'}}
        changes=[{'object_geometry_sha256':'a'*64},
                 {'object_geometry_sha256':'b'*64,'object_asset':{'nested':['新增',-0.0]}},
                 {'object_geometry_sha256':'c'*64,'source_physical_evidence':{'new':None},'label_native_binding':{}}]
        original_sha=hashlib.sha256;encoded={}
        for overrides in changes:
            chunks=[]
            class Recorder:
                def __init__(self):self.value=original_sha()
                def update(self,value):chunks.append(value);self.value.update(value)
                def hexdigest(self):return self.value.hexdigest()
            with patch.object(cb.hashlib,'sha256',side_effect=Recorder):
                actual=cb._canonical_alias_digest(shared,overrides,encoded)
            expected=canonical_bytes({**shared,**overrides})
            self.assertEqual(b''.join(chunks),expected)
            self.assertEqual(actual,original_sha(expected).hexdigest())

    def test_cached_field_does_not_replace_an_alias_override(self):
        shared={'object_geometry_sha256':'a'*64,'object_asset':{'nested':['original']}}
        cache={}
        first={'object_geometry_sha256':'a'*64}
        second={'object_geometry_sha256':'b'*64,'object_asset':{'nested':['replacement']}}
        self.assertEqual(cb._canonical_alias_digest(shared,first,cache),digest({**shared,**first}))
        self.assertEqual(cb._canonical_alias_digest(shared,second,cache),digest({**shared,**second}))
        self.assertEqual(cb._canonical_alias_digest(shared,first,cache),digest({**shared,**first}))

    def test_non_json_python_key_behavior_and_nonfinite_rejection_match_encoder(self):
        shared={2:{'value':-0.0},3:[1,None]}
        self.assertEqual(cb._canonical_alias_digest(shared,{4:'four'},{}),digest({**shared,4:'four'}))
        for invalid in (float('nan'),float('inf'),float('-inf')):
            with self.assertRaises(ValueError):cb._canonical_alias_digest({'mesh':[invalid]}, {}, {})
        # A replaced value is absent from the canonical object and must not be encoded.
        self.assertEqual(cb._canonical_alias_digest({'object_asset':float('nan')},{'object_asset':'valid'},{}),digest({'object_asset':'valid'}))

    def test_shared_geometry_is_encoded_once_but_every_alias_hash_is_checked(self):
        mesh=[{'vertices':[[0.,1.,2.]],'faces':[[0,0,0]]}]
        shared={'object_geometry_sha256':'a'*64,'collision_meshes':mesh}
        recipe=compact([shared],[(0,{'object_geometry_sha256':c*64}) for c in 'abc'])
        with patch.object(cb,'canonical_bytes',wraps=canonical_bytes) as encoder:
            expanded=cb.expand_native_recipe_assets(recipe)
        self.assertEqual(sum(call.args[0] is mesh for call in encoder.call_args_list),1)
        self.assertTrue(all(entry['collision_meshes'] is mesh for entry in expanded))
        recipe['object_geometry_aliases'][2]['expanded_recipe_entry_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'expanded recipe hash'):cb.expand_native_recipe_assets(recipe)

    def test_new_call_rechecks_changed_shared_geometry_and_overrides(self):
        shared={'object_geometry_sha256':'a'*64,'collision_meshes':[{'vertices':[[0.,1.,2.]],'faces':[]}]}
        recipe=compact([shared],[(0,{'object_geometry_sha256':'a'*64}),(0,{'object_geometry_sha256':'b'*64,'object_asset':'second'})])
        cb.expand_native_recipe_assets(recipe)
        for mutate in (lambda x:x['assets'][0]['collision_meshes'][0]['vertices'][0].__setitem__(0,1.),
                       lambda x:x['object_geometry_aliases'][1]['overrides'].__setitem__('object_asset','changed')):
            bad=deepcopy(recipe);mutate(bad)
            with self.assertRaisesRegex(ValueError,'expanded recipe hash'):cb.expand_native_recipe_assets(bad)
        bad=deepcopy(recipe);bad['object_geometry_aliases'][1]['overrides']['physical_profile']={'mass':2.}
        bad['object_geometry_aliases'][1]['expanded_recipe_entry_sha256']=digest({**bad['assets'][0],**bad['object_geometry_aliases'][1]['overrides']})
        with self.assertRaisesRegex(ValueError,'may not override'):cb.expand_native_recipe_assets(bad)

    def test_every_allowed_override_preserves_native_equivalence(self):
        shared={'object_geometry_sha256':'a'*64,'native_asset':'native.usda','native_asset_sha256':'b'*64,
                'physical_profile':{'mass':1.,'com':[0,0,0],'inertia':[[1,0,0],[0,1,0],[0,0,1]],'friction':.5},
                'exact_scale':[1,1,1],'mesh_to_object':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                'collision_meshes':[{'vertices':[[0.,0.,0.]],'faces':[]}], 'distance_recipe':{'grid_resolution':64}}
        root=Path.cwd();expected=cb.native_asset_equivalence(shared,root)
        for key,value in {'object_geometry_sha256':'c'*64,'object_asset':'other',
                          'source_physical_evidence':{'changed':'证据'},'label_native_binding':{'changed':True}}.items():
            self.assertEqual(cb.native_asset_equivalence({**shared,key:value},root),expected)


class CompactHeaderTests(unittest.TestCase):
    setUp=fixtures.CuratedBundleTests.setUp

    def prepare_compact_alias_bundle(self):
        second={'shapes':[{'kind':'cube','size':[.2,.2,.2]}],'basis':'second_identity'}
        geometry=self.root/'second_geometry.json';geometry.write_bytes(canonical_bytes(second));label_hash=digest(second)
        source=self.root/'clip1';metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
        metadata.update(object_geometry_path=str(geometry),object_geometry_sha256=label_hash)
        (source/'metadata.json').write_bytes(canonical_bytes(metadata))
        success=json.loads((source/'SUCCESS.json').read_text(encoding='utf-8'));success['metadata_sha256']=sha(source/'metadata.json')
        (source/'SUCCESS.json').write_bytes(canonical_bytes(success))
        records=json.loads((self.root/'metadata.json').read_text(encoding='utf-8'))
        records[1].update(object_geometry_path=str(geometry),object_geometry_sha256=label_hash,metadata_sha256=success['metadata_sha256'])
        metadata_path=self.root/'alias_metadata.json';metadata_path.write_bytes(canonical_bytes(records))
        split=self.root/'alias_split/split.json';write_split(self.selection,metadata_path,split.parent)
        recipe=json.loads(self.recipe.read_text(encoding='utf-8'));shared=recipe['assets'][0]
        recipe.update(compact([shared],[(0,{'object_geometry_sha256':self.geometry_sha}),
                                        (0,{'object_geometry_sha256':label_hash,'object_asset':'second'})]))
        self.recipe.write_bytes(canonical_bytes(recipe));output=self.root/'compact'
        bundle=cb.prepare_curated_bundle(split,self.recipe,output,self.kin)
        return output/'bundle.json',bundle,recipe

    def reseal(self,path,bundle):
        bundle['bundle_content_sha256']=digest({k:v for k,v in bundle.items() if k!='bundle_content_sha256'})
        path.write_bytes(canonical_bytes(bundle))

    def test_header_computes_one_equivalence_and_reuses_checked_entry_hashes(self):
        path,bundle,_=self.prepare_compact_alias_bundle()
        with patch.object(cb,'native_asset_equivalence',wraps=cb.native_asset_equivalence) as equivalence, \
                patch.object(cb,'digest',wraps=digest) as digest_spy:
            self.assertEqual(cb.read_curated_bundle_header(path),bundle)
        self.assertEqual(equivalence.call_count,1)
        # Full alias entries were hashed during expansion, never serialized again.
        self.assertFalse(any('object_geometry_sha256' in call.args[0] and 'collision_meshes' in call.args[0]
                             for call in digest_spy.call_args_list))

    def test_resealed_changed_binding_is_rejected_after_successful_read(self):
        path,bundle,_=self.prepare_compact_alias_bundle();cb.read_curated_bundle_header(path)
        for key in ('recipe_entry_sha256','native_equivalence_sha256'):
            bad=deepcopy(bundle);bad['object_geometry_bindings'][1][key]='0'*64;self.reseal(path,bad)
            with self.assertRaisesRegex(ValueError,'alias physics/geometry'):cb.read_curated_bundle_header(path)
        bad=deepcopy(bundle);bad['assets'][0]['object_geometry_aliases'].reverse();self.reseal(path,bad)
        with self.assertRaisesRegex(ValueError,'alias membership'):cb.read_curated_bundle_header(path)

    def test_new_call_keeps_recipe_and_split_byte_guards(self):
        path,_,_=self.prepare_compact_alias_bundle();cb.read_curated_bundle_header(path)
        original=self.recipe.read_bytes();self.recipe.write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError,'recipes changed'):cb.read_curated_bundle_header(path)
        self.recipe.write_bytes(original)
        bundle=json.loads(path.read_text(encoding='utf-8'));split=Path(bundle['manifest']['path'])
        split.write_bytes(split.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'split bytes changed'):cb.read_curated_bundle_header(path)

    def test_resealed_forbidden_physics_override_is_still_rejected(self):
        path,bundle,recipe=self.prepare_compact_alias_bundle();cb.read_curated_bundle_header(path)
        alias=recipe['object_geometry_aliases'][1];alias['overrides']['physical_profile']={**recipe['assets'][0]['physical_profile'],'mass':2.}
        alias['expanded_recipe_entry_sha256']=digest({**recipe['assets'][0],**alias['overrides']})
        self.recipe.write_bytes(canonical_bytes(recipe));bundle['recipes']['sha256']=sha(self.recipe);self.reseal(path,bundle)
        with self.assertRaisesRegex(ValueError,'may not override'):cb.read_curated_bundle_header(path)


if __name__=='__main__':unittest.main()
