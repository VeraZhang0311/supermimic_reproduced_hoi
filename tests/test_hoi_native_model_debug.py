"""Actual-model serializer tests; no native GPU execution is claimed."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hoi.v1.native_model_debug import export_actual_model, COLLISION_ARRAYS


class NativeModelDebugTests(unittest.TestCase):
    def test_real_compiled_mjb_and_portable_spec_roundtrip_without_model_mutation(self):
        import mujoco
        with tempfile.TemporaryDirectory(prefix="hoi-native-model-",dir="C:/Windows/Temp" if os.name=="nt" else None) as directory:
            root=Path(directory);out=root/"out";out.mkdir()
            mesh=root/"source.obj"
            mesh.write_text("v 0 0 0\nv .1 0 0\nv 0 .2 0\nv 0 0 .3\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n",encoding="utf8")
            source=root/"source.xml"
            source.write_text('<mujoco><asset><mesh name="piece" file="'+mesh.as_posix()+'"/></asset><worldbody><body name="Robot"><freejoint/><geom name="hull" type="mesh" mesh="piece"/></body></worldbody></mujoco>',encoding="utf8")
            spec=mujoco.MjSpec.from_file(str(source));model=spec.compile()
            original={key:np.array(getattr(model,key),copy=True) for key in COLLISION_ARRAYS}
            generated=out/"compiled_model.generated.xml"
            generated.write_text(spec.to_xml(),encoding="utf8")
            result=export_actual_model(model,out,generated)
            self.assertTrue(result["binary"]["collision_arrays_exact_roundtrip"])
            self.assertTrue(result["portable_xml"]["host_collision_arrays_agree"],result["portable_xml"])
            self.assertEqual(len(result["portable_xml"]["relocated_resources"]),1)
            self.assertEqual(result["physics_steps"],0)
            self.assertEqual(json.loads((out/"compiled_model_manifest.json").read_text())["mujoco_version"],mujoco.__version__)
            for key,before in original.items():
                np.testing.assert_array_equal(getattr(model,key),before,err_msg=key)
            # Portable source continues to load after the original asset moved.
            mesh.rename(root/"original-moved.obj")
            reloaded=mujoco.MjModel.from_xml_path(str(out/"compiled_model.xml"))
            self.assertEqual(reloaded.ngeom,model.ngeom)
            with self.assertRaises(FileExistsError):
                export_actual_model(model,out,generated)


if __name__=="__main__":
    unittest.main()

