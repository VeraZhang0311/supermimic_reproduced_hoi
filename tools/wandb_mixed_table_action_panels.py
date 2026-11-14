"""Add the six mixed table-action charts to the existing personal W&B view."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hoi_tables.table_actions import TABLE_ACTIONS, action_key


def sections(uid):
    result = []
    actions = [action_key(action) for action in TABLE_ACTIONS]
    for name, axis, prefix, charts in (
        ("Tables by Action - Training", "iteration", "tables_by_action",
         (("Normalized episode return", "normalized_return"), ("Episode failure rate", "failure_rate"), ("Completed episodes", "episode_count"))),
        ("Tables by Action - Protected Eval", "eval/iteration", "eval/tables20/per_action",
         (("Normalized clip return", "mean_normalized_return"), ("Clip failure rate", "failure_rate"), ("Evaluated clips", "clip_count"))),
    ):
        panels = []
        for index, (title, metric) in enumerate(charts):
            panels.append({"__id__": uid(name + title), "viewType": "Run History Line Plot", "isAuto": False,
                           "layout": {"x": index * 8, "y": 0, "w": 8, "h": 6},
                           "config": {"chartTitle": title, "xAxis": axis, "xAxisTitle": "Iteration" if axis == "iteration" else "Checkpoint",
                                      "metrics": [f"{prefix}/{action}/{metric}" for action in actions], "plotType": "line",
                                      "legendPosition": "south", "smoothingType": "none", "smoothingWeight": 0,
                                      "ignoreOutliers": False, "pointVisualizationMethod": "bucketing-gorilla"}})
        result.append({"__id__": uid(name), "name": name, "isOpen": False, "sorted": 0, "pinned": False,
                       "isPanelsAuto": False, "panels": panels, "flowConfig": {"snapToColumns": True, "columnsPerPage": 3, "rowsPerPage": 1},
                       "localPanelSettings": {"xAxis": axis, "smoothingType": "none", "smoothingWeight": 0}})
    return result


def main():
    import wandb
    from importlib.util import module_from_spec, spec_from_file_location

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    spec = spec_from_file_location("existing_wandb_view", args.workspace_helper)
    helper = module_from_spec(spec)
    spec.loader.exec_module(helper)
    api = wandb.Api(timeout=45)
    view = helper.get_view(api)
    original = json.loads(view["spec"])
    updated = json.loads(view["spec"])
    current = updated["section"]["panelBankConfig"]["sections"]
    additions = sections(helper.uid)
    names = {section["name"] for section in additions}
    current[:] = [section for section in current if section.get("name") not in names]
    hidden = next((i for i, section in enumerate(current) if section.get("name") == "Hidden Panels"), len(current))
    current[hidden:hidden] = additions
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "before.json").write_text(json.dumps(original, indent=2))
    result = {"applied": args.apply, "sections": sorted(names), "panel_count": 6}
    if args.apply:
        if helper.get_view(api)["spec"] != view["spec"]:
            raise RuntimeError("W&B workspace changed during preview")
        payload = dict(id=view["id"], entityName=helper.ENTITY, projectName=helper.PROJECT, name=view["name"],
                       displayName=view["displayName"], type=view["type"], spec=json.dumps(updated, separators=(",", ":")))
        change = api.client.execute(helper.MUTATION, payload)["upsertView"]
        if change["inserted"] or change["view"]["id"] != view["id"]:
            raise RuntimeError("W&B returned the wrong workspace")
        after = helper.get_view(api)["spec"]
        if json.loads(after) != updated:
            raise RuntimeError("W&B workspace readback differs")
        result["layout_sha256"] = hashlib.sha256(after.encode()).hexdigest()
        (args.output / "after.json").write_text(after)
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
