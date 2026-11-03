"""Mean and std over seeded runs of Camargo's next-activity results.

Usage (from the GenerativeLSTM folder): python summarize_camargo.py
Reads results/split_*/seed*/<dataset>/summary.csv and writes summary_camargo.csv.
std is the sample std (ddof=1) over runs and is empty when there is a single run.
Accuracy is reported two ways: mean over prefix lengths k (the way our STLFormer summaries are computed,
each k weighted equally) and over all predictions pooled.
"""
import glob, pathlib
import pandas as pd

root = pathlib.Path(".")
rows = []
for f in sorted(glob.glob("results/split_*/seed*/*/summary.csv")):
    p = pathlib.Path(f)
    d = pd.read_csv(f)
    d["variant"] = p.parts[1]
    rows.append(d)
if not rows:
    raise SystemExit("no results found under results/split_*/seed*/<dataset>/summary.csv")
res = pd.concat(rows, ignore_index=True)
metrics = ["accuracy_mean_over_k", "accuracy_over_all_predictions"]
out = (res.groupby(["variant", "dataset"])[metrics]
          .agg(["mean", "std"]).round(4))
out.columns = ["_".join(c) for c in out.columns]
out["runs"] = res.groupby(["variant", "dataset"]).size()
out = out.reset_index()
print(out.to_string(index=False))
out.to_csv("summary_camargo.csv", index=False)
print("written: summary_camargo.csv")
