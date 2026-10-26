from pathlib import Path
import hashlib
import json
import os


def verify():
    root = Path(os.environ['SUPERMIMIC_ROOT']).resolve()
    path = root / 'data/v11_staging.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == os.environ['EXPECTED_STAGE_SHA256']
    stage = json.loads(path.read_text())
    assert str(root) == stage['root']
    for name, digest in stage['files'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    from hoi.v1.config import source_identity
    assert source_identity() == stage['source_identity']
    from hoi.v1.train import read_config
    config, changes = read_config(root / 'configs/hoi_v1_1_observation_matching.json')
    assert not changes and config.to_dict() == stage['production_config']
    assert config.iterations == 20000 and config.num_envs == config.pi_batch == 4096 and config.horizon == 32
    from hoi.v1.observations import observation_contract
    contract = observation_contract(config.observation_profile)
    assert contract == stage["actor_observation_contract"]
    obj = contract["object_normalization"]
    assert (obj["history_dim"],obj["future_dim"],obj["epsilon"],obj["clip"]) == (9,63,.01,None)
    return root, stage
