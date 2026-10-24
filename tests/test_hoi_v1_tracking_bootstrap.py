"""Credentials remain process-local and cannot cross arbitrary endpoints."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hoi.v1.tracking_bootstrap import _netrc_key, tracking_sdk


class TrackingBootstrapTest(unittest.TestCase):
    def test_only_official_forge_endpoint_can_use_old_public_service_key(self):
        records = SimpleNamespace(hosts={'api.wandb.ai': ('api', None, 'secret-fixture'),
                                         'default': ('api', None, 'default-must-not-leak')})
        with patch('hoi.v1.tracking_bootstrap.netrc.netrc', return_value=records):
            self.assertEqual(_netrc_key('https://api.forge.coreweave.com'), 'secret-fixture')
            self.assertIsNone(_netrc_key('https://unrelated.example'))
            with self.assertRaises(ValueError):
                _netrc_key('https://api:embedded-secret@api.forge.coreweave.com')

    def test_key_and_import_path_restored_on_failure(self):
        previous_path = list(sys.path)
        fake = SimpleNamespace(init=lambda **kwargs: None, Settings=object)
        with patch.dict(os.environ, {}, clear=True), patch('hoi.v1.tracking_bootstrap._netrc_key', return_value='secret-fixture'), \
                patch('hoi.v1.tracking_bootstrap.importlib.import_module', return_value=fake):
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                with tracking_sdk('https://api.forge.coreweave.com') as sdk:
                    self.assertIs(sdk, fake)
                    self.assertEqual(os.environ['WANDB_API_KEY'], 'secret-fixture')
                    raise RuntimeError('intentional')
            self.assertNotIn('WANDB_API_KEY', os.environ)
            self.assertEqual(sys.path, previous_path)

    def test_existing_key_is_preserved_and_incomplete_overlay_rejected(self):
        fake = SimpleNamespace(init=lambda **kwargs: None, Settings=object)
        with patch.dict(os.environ, {'WANDB_API_KEY': 'existing-fixture'}, clear=True), \
                patch('hoi.v1.tracking_bootstrap._netrc_key') as lookup, \
                patch('hoi.v1.tracking_bootstrap.importlib.import_module', return_value=fake):
            with tracking_sdk('https://api.forge.coreweave.com'):
                self.assertEqual(os.environ['WANDB_API_KEY'], 'existing-fixture')
            lookup.assert_not_called()
            self.assertEqual(os.environ['WANDB_API_KEY'], 'existing-fixture')
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'HOI_WANDB_SDK_PATH': directory}):
            with self.assertRaisesRegex(ValueError, 'complete W&B SDK'):
                with tracking_sdk('https://api.forge.coreweave.com'):
                    self.fail('incomplete SDK must not be imported')
