"""Verify the compiled solver enables nonadjacent robot self-collision."""


def audit_self_collision(model):
    # World bodies are excluded. MuJoCo stores a single replicated template;
    # retain its parent/weld/exclude/mask rules, never force adjacent pairs on.
    excluded = set(map(int, model.exclude_signature))
    allowed = 0
    for a in range(model.ngeom):
        for b in range(a+1, model.ngeom):
            ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
            if not ba or not bb or ba == bb:
                continue
            wa, wb = int(model.body_weldid[ba]), int(model.body_weldid[bb])
            if wa == wb or int(model.body_parentid[wa]) == wb or int(model.body_parentid[wb]) == wa:
                continue
            if ((min(ba, bb) << 16) + max(ba, bb)) in excluded:
                continue
            if ((int(model.geom_contype[a]) & int(model.geom_conaffinity[b])) |
                    (int(model.geom_contype[b]) & int(model.geom_conaffinity[a]))):
                allowed += 1
    if not allowed:
        raise RuntimeError("compiled solver has no nonadjacent robot self-collision pairs")
    return {"enabled": True, "allowed_nonadjacent_geom_pairs": allowed}
