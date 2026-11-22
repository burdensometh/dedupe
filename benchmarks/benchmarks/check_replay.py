"""Assert that replaying a saved split reproduces its validation inputs.

Used by CI after ``benchmarks.entity_validation --save-plan`` and
``--load-plan``.  The held-out coverage from the replayed run must be
identical to the original split; otherwise a retrained model is being scored
on different validation records than the one the split was defined for.
"""

from __future__ import annotations

import json
import sys


def main(argv):
    original_path, replay_path = argv
    with open(original_path) as f:
        original = json.load(f)
    with open(replay_path) as f:
        replay = json.load(f)

    left = original["dedupe"]["coverage"]
    right = replay["dedupe_replay"]["coverage"]
    if left != right:
        print("validation inputs changed between save and replay", file=sys.stderr)
        print(f"original: {left}", file=sys.stderr)
        print(f"replay:   {right}", file=sys.stderr)
        return 1

    truth = replay["dedupe_replay"]["cross_validation"]["aggregate"]["n_truth"]
    print(f"replay reproduced validation inputs ({truth} held-out pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
