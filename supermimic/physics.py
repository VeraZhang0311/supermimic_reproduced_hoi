"""Shared, explicit robot collision settings; importable without the simulator."""
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import hashlib
import json
import os
import uuid

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PhysicsSettings:
    hand_collision: bool = True
    self_collision: bool = True

    def flags(self):
        return ['--hand-collision' if self.hand_collision else '--no-hand-collision',
                '--self-collision' if self.self_collision else '--no-self-collision']

    def export(self):
        for key, value in asdict(self).items():
            os.environ['SUPERMIMIC_' + key.upper()] = '1' if value else '0'
        return self

    @classmethod
    def environment(cls):
        values = {}
        for key in ('hand_collision', 'self_collision'):
            value = os.environ.get('SUPERMIMIC_' + key.upper(), '1')
            if value not in ('0', '1'):
                raise ValueError('SUPERMIMIC_' + key.upper() + ' must be 0 or 1')
            values[key] = value == '1'
        return cls(**values)


def add_physics_arguments(parser):
    for name in ('hand', 'self'):
        parser.add_argument('--' + name + '-collision', dest=name + '_collision',
                            action=argparse.BooleanOptionalAction, default=None,
                            help='Enable ' + name + ' collision (default: enabled for new runs).')


def resolve_arguments(args, saved=None, require_legacy_explicit=False):
    """Inherit new saved settings; old checkpoints require both explicit switches."""
    saved = saved or {}
    keys = ('hand_collision', 'self_collision')
    explicit = {key for key in keys if getattr(args, key) is not None}
    if require_legacy_explicit and not all(key in saved for key in keys) and explicit != set(keys):
        raise ValueError('Old checkpoint has no collision metadata. Specify both '
                         '--[no-]hand-collision and --[no-]self-collision explicitly.')
    values = {key: getattr(args, key) if key in explicit else saved.get(key, True) for key in keys}
    if not all(type(value) is bool for value in values.values()):
        raise ValueError('Saved collision settings must be booleans')
    for key, value in values.items():
        setattr(args, key, value)
    return PhysicsSettings(**values).export(), explicit


def validate_resume(saved, settings, explicit):
    values = asdict(settings)
    if all(key in saved for key in values):
        if any(saved[key] != value for key, value in values.items()):
            raise ValueError('Resume collision settings differ from the saved checkpoint; '
                             'use a separate evaluation for a physics comparison.')
    elif set(values) != explicit:
        raise ValueError('Old resume checkpoint has unknown collision settings; specify both collision switches.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_hand_asset(path):
    from hoi.hand_collision import HAND_SHA, RECIPE
    path = Path(path)
    report = json.loads((path.parent / 'hand-asset.json').read_text())
    if sha(path) != report['usd_sha256'] or sha(report['source_usd']) != report['source_usd_sha256']:
        raise ValueError('Hand collision asset or source USD changed')
    if report['recipe'] != RECIPE or not report['original_authored_values_unchanged']:
        raise ValueError('Unexpected hand recipe or original robot mutation')
    for side, digest in HAND_SHA.items():
        if report['hands'][side]['stl_sha256'] != digest:
            raise ValueError('Unexpected hand geometry')
    return report


def configure_robot(robot, root=None, output=None, settings=None):
    """Configure an instance, never the shared G1_CFG or an existing source USD."""
    root = Path(root or ROOT)
    settings = settings or PhysicsSettings.environment()
    base = root / 'data/g1_usd/g1_29dof/g1_29dof.usda'
    current = Path(robot.spawn.usd_path)
    asset = None
    if settings.hand_collision:
        if current.resolve() != base.resolve():
            asset = load_hand_asset(current)
        else:
            from hoi.hand_collision import prepare
            output = output or os.environ.get('SUPERMIMIC_ASSET_DIR')
            output = Path(output).resolve() if output else root / '.cache/robot-assets' / uuid.uuid4().hex
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                asset = load_hand_asset(output / 'g1_rubber_hand_collision.usda')
            else:
                asset = prepare(root, output, None)
            robot.spawn.usd_path = str(output / 'g1_rubber_hand_collision.usda')
    elif current.resolve() != base.resolve():
        raise ValueError('A robot USD override cannot be combined with --no-hand-collision')
    robot.spawn.articulation_props.enabled_self_collisions = settings.self_collision
    return {**asdict(settings), 'robot_usd': robot.spawn.usd_path,
            'robot_usd_sha256': sha(robot.spawn.usd_path), 'hand_asset': asset}


def compiled_self_collision_audit(model, expected):
    """Check actual compiled MuJoCo pairs, excluding objects/ground/welds."""
    import mujoco
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or '' for i in range(model.nbody)]
    objects = {i for i, name in enumerate(names) if '_Object' in name or '/Object' in name}
    excluded = set(map(int, model.exclude_signature))
    allowed = 0
    for a in range(model.ngeom):
        for b in range(a + 1, model.ngeom):
            ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
            if not ba or not bb or ba == bb or ba in objects or bb in objects:
                continue
            wa, wb = int(model.body_weldid[ba]), int(model.body_weldid[bb])
            if wa == wb or int(model.body_parentid[wa]) == wb or int(model.body_parentid[wb]) == wa:
                continue
            if ((min(ba, bb) << 16) + max(ba, bb)) in excluded:
                continue
            if ((int(model.geom_contype[a]) & int(model.geom_conaffinity[b])) |
                    (int(model.geom_contype[b]) & int(model.geom_conaffinity[a]))):
                allowed += 1
    if bool(allowed) != expected:
        raise ValueError(f'Compiled self collision differs from requested {expected}: {allowed} pairs')
    return {'enabled': expected, 'allowed_nonadjacent_robot_geom_pairs': allowed}
