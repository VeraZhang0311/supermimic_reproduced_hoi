"""Isolated SDK/credential setup; never serialize or print authentication."""
from contextlib import contextmanager
import importlib
import netrc
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit


def _netrc_key(base_url):
    endpoint = urlsplit(base_url)
    if (endpoint.scheme not in ('https', 'http') or not endpoint.hostname or
            endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError('invalid W&B authentication endpoint')
    hosts = [endpoint.hostname]
    # The repository's official Forge endpoint supersedes the old W&B hostname.
    # Never send an unrelated endpoint the public-service credential.
    if endpoint.hostname == 'api.forge.coreweave.com':
        hosts.append('api.wandb.ai')
    try:
        credentials = netrc.netrc()
        for host in hosts:
            # netrc.authenticators would also return a machine-default key.
            # Authentication here must bind an explicit, expected hostname.
            entry = credentials.hosts.get(host)
            if entry and entry[2]:
                return entry[2]
    except (OSError, netrc.NetrcParseError):
        pass
    return None


@contextmanager
def tracking_sdk(base_url):
    """Use a tracking-only overlay and existing credentials for this process.

    Call after simulator workers have spawned. The explicit Slurm exports must
    contain only the non-secret overlay path/destination, never WANDB_API_KEY.
    """
    previous_path = list(sys.path)
    previous_key = os.environ.get('WANDB_API_KEY')
    try:
        overlay = os.environ.get('HOI_WANDB_SDK_PATH')
        if overlay:
            overlay = Path(overlay).resolve(strict=True)
            if not (overlay / 'wandb/__init__.py').is_file():
                raise ValueError('tracking overlay must contain a complete W&B SDK')
            loaded = sys.modules.get('wandb')
            if loaded is not None and getattr(loaded, '__file__', None):
                origin = Path(loaded.__file__).resolve()
                if overlay not in origin.parents:
                    raise RuntimeError('W&B was already imported from a different SDK runtime')
            elif loaded is not None:
                # A stripped runtime can leave an empty namespace package.
                del sys.modules['wandb']
            sys.path.insert(0, str(overlay))
        if not previous_key:
            credential = _netrc_key(base_url)
            if credential:
                os.environ['WANDB_API_KEY'] = credential
        sdk = importlib.import_module('wandb')
        if not callable(getattr(sdk, 'init', None)) or not hasattr(sdk, 'Settings'):
            raise RuntimeError('a complete W&B SDK is required for explicitly enabled online tracking')
        yield sdk
    finally:
        sys.path[:] = previous_path
        if previous_key is None:
            os.environ.pop('WANDB_API_KEY', None)
        else:
            os.environ['WANDB_API_KEY'] = previous_key
