"""Entity-level cross-validation on the canonical restaurant datasets.

The restaurant data ships with a ``unique_id`` column: records that share a
``unique_id`` are the same restaurant.  A record-level split would let the
same restaurant appear in both training and validation, which overstates
generalisation.  This script instead:

1. resolves restaurant identities from ``unique_id``;
2. assigns whole restaurants to folds;
3. builds per-fold training pairs and distances (never using held-out
   restaurants);
4. scores the real blocking candidates on held-out records;
5. reports split coverage and held-out link quality for every fold.

The report is written as JSON so validation inputs stay reproducible: run
with the same ``--seed`` and ``--folds`` to replay the exact split, and save
the plan next to the report with ``--save-plan``.
"""

from __future__ import annotations

import argparse
import json
import sys

import dedupe
from benchmarks import common
from dedupe import validation


def load_restaurant(pathlike):
    """Load a restaurant CSV into a dedupe data dict, preserving ``unique_id``."""
    import csv

    data = {}
    with open(pathlike) as f:
        for i, row in enumerate(csv.DictReader(f)):
            clean_row = {}
            for key, value in row.items():
                cleaned = common.pre_process(value)
                clean_row[key] = cleaned
            data[str(pathlike) + str(i)] = clean_row
    return data


def dedupe_variables():
    return [
        dedupe.variables.String("name"),
        dedupe.variables.Exact("name"),
        dedupe.variables.String("address"),
        dedupe.variables.ShortString("cuisine", has_missing=True),
        dedupe.variables.ShortString("city"),
    ]


def link_variables():
    return [
        dedupe.variables.String("name"),
        dedupe.variables.String("address"),
        dedupe.variables.String("cuisine"),
        dedupe.variables.String("city"),
    ]


def run_dedupe(args, report):
    data = load_restaurant(common.DATASETS_DIR / "restaurant-nophone-training.csv")
    plan = validation.EntityPlan.from_data(
        data, entity_key="unique_id", n_folds=args.folds, seed=args.seed
    )
    report["dedupe"] = {
        "n_records": len(data),
        "n_entities": len(set(plan.entity_of.values())),
        "coverage": plan.coverage(),
        "cross_validation": validation.run_dedupe_cv(
            plan,
            dedupe_variables(),
            threshold=args.threshold,
            n_distinct=args.n_distinct,
            num_cores=args.num_cores,
        ).to_dict(),
    }
    if args.save_plan:
        plan.save(args.save_plan + ".dedupe.json")
    return plan


def run_link(args, report):
    data_1 = load_restaurant(common.DATASETS_DIR / "restaurant-1.csv")
    data_2 = load_restaurant(common.DATASETS_DIR / "restaurant-2.csv")
    plan = validation.LinkPlan.from_data(
        data_1, data_2, entity_key="unique_id", n_folds=args.folds, seed=args.seed
    )
    report["link"] = {
        "n_records_1": len(data_1),
        "n_records_2": len(data_2),
        "n_entities": len(set(plan.entity_of.values())),
        "coverage": plan.coverage(),
        "cross_validation": validation.run_link_cv(
            plan,
            link_variables(),
            threshold=args.threshold,
            constraint=args.constraint,
            n_distinct=args.n_distinct,
            num_cores=args.num_cores,
        ).to_dict(),
    }
    if args.save_plan:
        plan.save(args.save_plan + ".link.json")
    return plan


def replay(args, report):
    """Retrain on all entities, then replay a saved split for validation."""
    data = load_restaurant(common.DATASETS_DIR / "restaurant-nophone-training.csv")
    plan = validation.EntityPlan.load(args.load_plan, data)
    matcher = dedupe.Dedupe(dedupe_variables(), num_cores=args.num_cores)
    validation.refit_full_model(
        matcher,
        data,
        entity_key="unique_id",
        sample_size=len(data),
        n_distinct=args.n_distinct,
        seed=args.seed,
    )
    report["dedupe_replay"] = {
        "coverage": plan.coverage(),
        "cross_validation": validation.replay_dedupe(
            plan, matcher, threshold=args.threshold
        ).to_dict(),
    }
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-distinct", type=int, default=2000)
    parser.add_argument("--num-cores", type=int, default=0)
    parser.add_argument(
        "--constraint",
        default="one-to-one",
        choices=["one-to-one", "many-to-one", "many-to-many"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["dedupe", "link"],
        choices=["dedupe", "link"],
    )
    parser.add_argument("--save-plan", default=None)
    parser.add_argument(
        "--load-plan",
        default=None,
        help="replay a saved plan against a fully retrained model "
        "instead of building a new split",
    )
    parser.add_argument("--output", default=None)

    args = parser.parse_args(argv)

    report = {"seed": args.seed, "folds": args.folds, "threshold": args.threshold}
    if args.load_plan:
        replay(args, report)
    else:
        if "dedupe" in args.modes:
            run_dedupe(args, report)
        if "link" in args.modes:
            run_link(args, report)

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w") as f:
            f.write(payload)
    print(payload)
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
