"""
neuromesh/cli.py -- unified command-line interface.

Design choice: each subcommand's full flag set already lives in its
underlying script's own argparse parser (train_real.py, evaluate_frozen_test.py,
mechanism_analysis.py, compare.py). Rather than duplicate those flags here
(and risk them drifting out of sync), this CLI is a thin router: it reads the
subcommand name, then forwards the remaining argv verbatim to that script's
own main(). Run `neuromesh <command> --help` to see the real flags.
"""
import sys


COMMANDS = ["train", "evaluate", "analyze-controller", "compare", "reproduce"]


def _forward(module_path, entry_name, rest):
    import importlib
    mod = importlib.import_module(module_path)
    sys.argv = [f"neuromesh-{entry_name}"] + rest
    mod.main()


def _print_reproduce_instructions():
    print("""\
NeuroMesh v1 pilot -- exact commands used to produce the frozen/validation results
====================================================================================

These are recorded for provenance, not auto-executed (they need the real
BraTS-derived dataset, which this repository does not ship -- see
docs/dataset.md). Dataset paths below are placeholders for your local copy.

1. NeuroMesh (4 epochs, protocol-frozen v1):

   neuromesh train --dataset brats_h5 \\
       --data-root <path-to-h5-files> --metadata-csv <path-to-metadata.csv> \\
       --model neuromesh --base-ch 16 --image-size 96 --batch-size 8 --epochs 4 \\
       --val-fraction 0.15 --test-fraction 0.15 --split-seed 42 \\
       --run-name neuromesh_real30

2. Baselines (identical protocol, swap --model and --run-name):

   neuromesh train --dataset brats_h5 --data-root <...> --metadata-csv <...> \\
       --model plain        --base-ch 16 --image-size 96 --batch-size 8 --epochs 4 \\
       --val-fraction 0.15 --test-fraction 0.15 --split-seed 42 --run-name plain_real30

   neuromesh train ... --model dropout      --run-name dropout_real30
   neuromesh train ... --model static_gate  --run-name staticgate_real30

3. Frozen test evaluation (run ONCE per model -- see PROTOCOL_FREEZE_v1.md):

   neuromesh evaluate --checkpoint checkpoints/neuromesh_real30.pt \\
       --split test --results-out results/frozen/FROZEN_TEST_neuromesh_real30.json

4. Validation evaluation (safe to re-run during development):

   neuromesh evaluate --checkpoint <ckpt> --split val --results-out <out.json>

5. Mechanism analysis (mask/gate/topology, per condition):

   neuromesh analyze-controller --checkpoint <ckpt> --out-dir <dir> --split val --condition clean
   # repeat --condition for each of: FLAIR_missing T1_missing T1ce_missing T2_missing
   neuromesh analyze-controller --checkpoint <ckpt> --out-dir <dir> --split val   # PASS 2, representative slices

6. Regenerate tables from stored results (no new computation):

   neuromesh compare --results-dir . --out-dir results/manifests

Full experimental protocol (split seed, patient IDs, exact hyperparameters,
what was and wasn't touched on the test set): see PROTOCOL_FREEZE_v1.md and
docs/experiments.md.
""")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(f"neuromesh -- research CLI. Commands: {', '.join(COMMANDS)}")
        print("Run `neuromesh <command> --help` for that command's real flags.")
        return

    command, rest = sys.argv[1], sys.argv[2:]

    if command == "train":
        _forward("neuromesh.training.train_real", "train", rest)
    elif command == "evaluate":
        _forward("neuromesh.evaluation.evaluate_frozen_test", "evaluate", rest)
    elif command == "analyze-controller":
        _forward("neuromesh.analysis.mechanism_analysis", "analyze-controller", rest)
    elif command == "compare":
        _forward("neuromesh.analysis.compare", "compare", rest)
    elif command == "reproduce":
        _print_reproduce_instructions()
    else:
        print(f"Unknown command '{command}'. Commands: {', '.join(COMMANDS)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
