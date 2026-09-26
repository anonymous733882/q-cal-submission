#!/usr/bin/env python3
"""Export one policy from a held-out gradient suite as a policy JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite_json", required=True)
    p.add_argument("--policy_name", default=None)
    p.add_argument("--out_json", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    suite_path = Path(args.suite_json)
    suite = json.loads(suite_path.read_text())
    policy_name = args.policy_name
    if policy_name is None:
        names = suite.get("global_policy_names") or []
        if not names:
            raise KeyError("suite has no global_policy_names; pass --policy_name")
        policy_name = str(names[0])
    policy = suite.get("policies", {}).get(policy_name)
    if not isinstance(policy, dict):
        raise KeyError(f"policy {policy_name!r} not found in {suite_path}")
    stage1_weight_s = os.environ.get("DUALSIGNAL_STAGE1_WEIGHT")
    stage1_weight = float(stage1_weight_s) if stage1_weight_s is not None else None
    out = {
        "policy": policy,
        "source_suite": str(suite_path),
        "source_policy_name": policy_name,
        "source_suite_plan": suite.get("plan", {}),
        "protocol": {
            "stage1_shallow_weight": stage1_weight,
            "stage1_vit_weight": 1.0 - stage1_weight if stage1_weight is not None else None,
        },
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"wrote": str(out_path), "policy_name": policy_name}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
