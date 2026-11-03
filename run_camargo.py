"""Train and evaluate Camargo et al.'s GenerativeLSTM (next activity) on one of our logs, seeded.

Usage (from the GenerativeLSTM folder):
  python run_camargo.py --name helpdesk --csv data/helpdesk/helpdesk.csv --seed 42
"""
import os, sys, argparse, random, json, glob, shutil, time
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--search", action="store_true", help="use the repo's full hyperopt search space (model_type, n_size, l_size, lstm_act, norm_method) instead of the fixed config")
ap.add_argument("--max_eval", type=int, default=1, help="number of hyperopt trials when --search is on")
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--model_type", default="shared_cat")
ap.add_argument("--n_size", type=int, default=5)
ap.add_argument("--l_size", type=int, default=100)
ap.add_argument("--lstm_act", default="tanh")
ap.add_argument("--norm_method", default="lognorm")
ap.add_argument("--result_dir", default="results")
ap.add_argument("--split", choices=["timeline", "stl"], default="timeline",
                help="timeline: the repo's own 80/20 split by time. stl: the same test cases as STLFormer/MTLFormer "
                     "(last 20%% of the cases in the order of --stl_csv)")
ap.add_argument("--stl_csv", default=None, help="the CSV STLFormer used (needed for --split stl)")
ap.add_argument("--val_inside_train", action="store_true",
                help="validate (early stopping, checkpoint choice) on the last 20%% of the TRAINING part only. "
                     "As released, the repo validates on the last 20%% of the whole log, i.e. on the test region.")
ap.add_argument("--tag", default="", help="suffix for the results folder, e.g. n15")
ap.add_argument("--lstm_dropout", type=float, default=0.2, help="dropout of every LSTM layer; the repository uses 0.2. With Keras 3 this makes the model collapse to one class at inference on some logs (BPIC2012_W), 0.0 avoids it")
ap.add_argument("--legacy_sort", action="store_true", help="keep pandas' default (unstable) sort_values, as in the repo; reorders events with identical timestamps inside a case")
args = ap.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CAMARGO_DROPOUT"] = str(args.lstm_dropout)
os.environ["PYTHONHASHSEED"] = str(args.seed)
os.environ["HYPEROPT_FMIN_SEED"] = str(args.seed)   # hyperopt trial sampling would otherwise be unseeded
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
random.seed(args.seed)
import numpy as np
np.random.seed(args.seed)
import pandas as pd
if not args.legacy_sort:
    # pandas' default sort (quicksort) is not stable: events of one case with identical timestamps (48.6% of Helpdesk events,
    # ~10-12% of BPIC2012 A/O, date-granular logs) get shuffled by the repo's log.sort_values(timestamp) calls. Make every sort stable.
    _orig_sort_values = pd.DataFrame.sort_values
    def _stable_sort_values(self, *a, **kw):
        kw.setdefault("kind", "stable")
        return _orig_sort_values(self, *a, **kw)
    pd.DataFrame.sort_values = _stable_sort_values
import tensorflow as tf
tf.random.set_seed(args.seed)
import keras
keras.utils.set_random_seed(args.seed)
tf.config.experimental.enable_op_determinism()

# ---- 1. convert our csv (Case ID, Activity, Complete Timestamp) to the format the repo reads
raw = pd.read_csv(args.csv)
raw["Complete Timestamp"] = pd.to_datetime(raw["Complete Timestamp"], format="mixed", dayfirst=False)
if "Resource" not in raw.columns:
    raw["Resource"] = "resource"          # our logs carry no resource column: a single role
raw["Complete Timestamp"] = raw["Complete Timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
os.makedirs("input_files", exist_ok=True)
fname = args.name + ".csv"
raw[["Case ID", "Activity", "Resource", "Complete Timestamp"]].to_csv(os.path.join("input_files", fname), index=False)
for f in glob.glob(os.path.join("input_files", "embedded_matix", "*" + args.name + ".emb")):
    os.remove(f)                           # embeddings are retrained with this seed

# ---- 2. training (hyperparameters fixed, no Bayesian search)
from model_training import model_trainer as tr
column_names = {"Case ID": "caseid", "Activity": "task", "Resource": "user"}
params = dict()
params["one_timestamp"] = True
params["read_options"] = {"timeformat": "mixed", "column_names": column_names,
                          "one_timestamp": True}
params["val_inside_train"] = args.val_inside_train or args.split == "stl"
params.update(file_name=fname, opt_method="bayesian", max_eval=args.max_eval if args.search else 1, rp_sim=0.85,
              batch_size=args.batch_size, imp=2, epochs=args.epochs, dense_act=["linear"], optim=["Nadam"])
if args.search:   # exactly the search space of the original dg_training.py (default, non-cx, LSTM variants)
    params.update(norm_method=["max", "lognorm"], n_size=[5, 10, 15], l_size=[50, 100],
                  lstm_act=["selu", "tanh"], model_type=["shared_cat", "concatenated"])
else:
    params.update(norm_method=[args.norm_method], n_size=[args.n_size], l_size=[args.l_size],
                  lstm_act=[args.lstm_act], model_type=[args.model_type])
if args.split == "stl":
    assert args.stl_csv, "--split stl needs --stl_csv"
    order = pd.read_csv(args.stl_csv)["Case ID"].astype(str).drop_duplicates().tolist()
    test_cases = set(order[int(len(order) * 0.8):])
    def split_by_cases(self, size, one_ts):
        """Same train/test cases as STLFormer: first 80% of the cases in file order train, last 20% test."""
        is_test = self.log.caseid.astype(str).isin(test_cases)
        self.log_test = self.log[is_test].sort_values("end_timestamp", ascending=True).reset_index(drop=True)
        self.log_train = self.log[~is_test].sort_values("end_timestamp", ascending=True).reset_index(drop=True)
    tr.ModelTrainer.split_timeline = split_by_cases
    # validation for early stopping: a random 20% of the training CASES (seeded), not the last 20% by time,
    # so that validation and test come from the same kind of cases
    from model_training import model_optimizer as op
    def val_by_cases(size, log, one_ts):
        ids = log.caseid.astype(str).drop_duplicates().tolist()
        np.random.RandomState(args.seed).shuffle(ids)
        in_train = log.caseid.astype(str).isin(set(ids[:int(len(ids) * size)]))
        key = "end_timestamp"
        return (log[~in_train].sort_values(key).reset_index(drop=True),
                log[in_train].sort_values(key).reset_index(drop=True))     # (validation, train)
    op.ModelOptimizer.split_timeline = staticmethod(val_by_cases)
before = set(os.listdir("output_files")) if os.path.isdir("output_files") else set()
os.makedirs("output_files", exist_ok=True)
t0 = time.time()
tr.ModelTrainer(params)
print(f"training done in {time.time()-t0:.0f}s", flush=True)
new = [d for d in set(os.listdir("output_files")) - before if os.path.isdir(os.path.join("output_files", d, "parameters"))]
assert len(new) == 1, f"expected one trained model folder, found {new}"
folder = new[0]

# ---- 3. next-event prediction on the held-out 20%
from model_prediction import model_predictor as pr
pp = dict()
pp["one_timestamp"] = True
pp["read_options"] = {"timeformat": "mixed", "column_names": column_names,
                      "one_timestamp": True, "filter_d_attrib": False}
pp.update(activity="predict_next", folder=folder, model_file=fname.replace(".csv", ".h5"),
          variant="arg_max", rep=1, is_single_exec=False)
pr.ModelPredictor(pp)
print("output folder:", folder)

# ---- 4. metrics from the predictions (the repo's own evaluation call is broken)
import ast, json
gen = pd.read_csv(os.path.join("output_files", folder, "gen_" + args.name + "_1.csv"))
gen["k"] = gen.ac_prefix.map(lambda s: len(ast.literal_eval(s)) - 1)   # real events in the prefix (the 'start' token is not counted)
gen = gen[gen.k >= 1].copy()                                           # same prefix set as STLFormer: at least one real event
gen["correct"] = (gen.ac_expect.astype(int) == gen.ac_pred.astype(int)).astype(float)
gen["abs_err_days"] = (gen.tm_expect - gen.tm_pred).abs() / 86400.0     # next-event time, seconds -> days
per_k = gen.groupby("k").agg(accuracy=("correct", "mean"), mae_next_time_days=("abs_err_days", "mean"),
                             n=("correct", "size")).reset_index()
best = json.load(open(os.path.join("output_files", folder, "parameters", "model_parameters.json")))
out_dir = os.path.join(args.result_dir, "split_" + args.split + (("_" + args.tag) if args.tag else ""), "seed%d" % args.seed, args.name)
os.makedirs(out_dir, exist_ok=True)
per_k.to_csv(os.path.join(out_dir, "results_per_k.csv"), index=False)
row = {"dataset": args.name, "seed": args.seed, "split": args.split, "val_inside_train": params["val_inside_train"],
       "accuracy_mean_over_k": per_k.accuracy.mean(),
       "accuracy_over_all_predictions": gen.correct.mean(),
       "next_time_mae_days_mean_over_k": per_k.mae_next_time_days.mean(),
       "next_time_mae_days_over_all_predictions": gen.abs_err_days.mean(),
       "n_predictions": len(gen), "train_seconds": round(time.time() - t0),
       "model_type": best.get("model_type"), "n_size": best.get("n_size"), "l_size": best.get("l_size"),
       "lstm_act": best.get("lstm_act"), "norm_method": best.get("norm_method"), "epochs_max": args.epochs,
       "search": args.search, "lstm_dropout": args.lstm_dropout, "max_eval": params["max_eval"], "model_folder": folder}
pd.DataFrame([row]).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
print(json.dumps(row, indent=1, default=float))
