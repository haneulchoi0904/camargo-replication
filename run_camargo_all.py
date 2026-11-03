"""Run Camargo et al.'s GenerativeLSTM (next activity) on several logs, several seeded runs each.

Run it from the GenerativeLSTM folder (the one that contains run_camargo.py):

  # as released (the repo's own 80/20 split by time), 3 runs with seeds 42, 43, 44
  python run_camargo_all.py --inputs_dir camargo_inputs --runs 3 --seed 42

  # on STLFormer's exact test cases (last 20% of the cases of the CSVs STLFormer used), one run, seed 42
  python run_camargo_all.py --inputs_dir camargo_inputs --split stl --stl_dir <folder with <name>/<name>.csv> --runs 1 --seed 42

Run r uses seed (seed + r - 1). Finished runs are skipped, so the command can be stopped and restarted.
"""
import argparse, pathlib, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=["helpdesk", "BPIC2012_A", "BPIC2012_O", "BPIC2012_W"])
ap.add_argument("--inputs_dir", default="camargo_inputs", help="folder with <name>_with_resource.csv (or <name>.csv / <name>/<name>.csv with --suffix .csv)")
ap.add_argument("--suffix", default="_with_resource.csv", help="input file suffix; use .csv to run on the shipped STLFormer/MTLFormer CSVs (no Resource column: one constant role)")
ap.add_argument("--runs", type=int, default=1)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--split", choices=["timeline", "stl"], default="timeline")
ap.add_argument("--stl_dir", default=None, help="folder holding <name>.csv (or <name>/<name>.csv) as used by STLFormer (for --split stl)")
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--val_inside_train", action="store_true")
ap.add_argument("--search", action="store_true", help="full hyperopt search space of the original repo")
ap.add_argument("--max_eval", type=int, default=10, help="hyperopt trials per run when --search is on")
ap.add_argument("--lstm_dropout", type=float, default=0.2, help="LSTM dropout (repository: 0.2)")
ap.add_argument("--tag", default="")
args = ap.parse_args()

here = pathlib.Path(__file__).resolve().parent
for name in args.datasets:
    cands = [pathlib.Path(args.inputs_dir) / f"{name}{args.suffix}", pathlib.Path(args.inputs_dir) / name / f"{name}{args.suffix}"]
    csv = next((c for c in cands if c.exists()), None)
    if csv is None:
        sys.exit(f"missing input, tried: {', '.join(map(str, cands))}")
    for r in range(1, args.runs + 1):
        seed = args.seed + r - 1
        folder = "split_" + args.split + (("_" + args.tag) if args.tag else "")
        done = here / "results" / folder / f"seed{seed}" / name / "summary.csv"
        if done.exists():
            print(f"skip {name} seed {seed} (already done)", flush=True)
            continue
        cmd = [sys.executable, "run_camargo.py", "--name", name, "--csv", str(csv), "--seed", str(seed),
               "--split", args.split, "--epochs", str(args.epochs)]
        if args.split == "stl":
            if not args.stl_dir:
                sys.exit("--split stl needs --stl_dir")
            c1 = pathlib.Path(args.stl_dir) / f"{name}.csv"          # data\\helpdesk.csv
            c2 = pathlib.Path(args.stl_dir) / name / f"{name}.csv"   # data\\helpdesk\\helpdesk.csv
            stl_csv = c1 if c1.exists() else c2
            if not stl_csv.exists():
                sys.exit(f"cannot find {c1} or {c2}")
            cmd += ["--stl_csv", str(stl_csv)]
        if args.search:
            cmd += ["--search", "--max_eval", str(args.max_eval)]
        if args.lstm_dropout != 0.2:
            cmd += ["--lstm_dropout", str(args.lstm_dropout)]
        if args.val_inside_train:
            cmd += ["--val_inside_train"]
        if args.tag:
            cmd += ["--tag", args.tag]
        print(">>", " ".join(cmd), flush=True)
        t = time.time()
        subprocess.run(cmd, cwd=here, check=True)
        print(f"   ({time.time() - t:.0f}s)", flush=True)
print("all done")
