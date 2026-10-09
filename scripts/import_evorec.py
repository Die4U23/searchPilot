"""把 EvoRec 结果 JSON 只读导入实验登记。SHA-256 不一致则拒绝。

用法::

    python scripts/import_evorec.py results.json \\
        --repo Die4U23/EvoRec --commit <sha> --run-id <id> \\
        --expect-sha256 <digest> --registry artifacts/experiments/registry.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from searchpilot.experiments.import_evorec import import_evorec_result
from searchpilot.experiments.store import InMemoryExperimentStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="EvoRec results JSON")
    parser.add_argument("--repo", required=True, help="EvoRec repository")
    parser.add_argument("--commit", required=True, help="EvoRec commit")
    parser.add_argument("--run-id", required=True, dest="run_id", help="EvoRec run id")
    parser.add_argument(
        "--expect-sha256",
        default=None,
        dest="expect_sha256",
        help="reject import when the file digest differs",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="optional InMemoryExperimentStore JSON; loaded then saved on success",
    )
    args = parser.parse_args(argv)

    store = (
        InMemoryExperimentStore.load(args.registry)
        if args.registry is not None
        else InMemoryExperimentStore()
    )
    try:
        record = import_evorec_result(
            args.path,
            store,
            repo=args.repo,
            commit=args.commit,
            run_id=args.run_id,
            expect_sha256=args.expect_sha256,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.registry is not None:
        store.save(args.registry)
    print(json.dumps(record.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
