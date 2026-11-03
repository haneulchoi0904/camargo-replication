# Camargo et al. (GenerativeLSTM) replication: next-activity accuracy

Part of a replication of Wang et al., "MTLFormer: Multi-task learning guided transformer network for business process prediction", IEEE Access 11 (2023) (course 2IMI00, TU/e). This repository reproduces the **Camargo et al. baseline, next-activity accuracy (Table 4 of the paper)** on Helpdesk and BPIC2012 A, O, W. The paper also reports Camargo et al. in Table 6 (remaining time); that is not replicated here.

This is a patched copy of GenerativeLSTM (Camargo et al., Apache 2.0, master branch downloaded 2026-10-19) with the changes listed below. 

## Environment
Tested on Windows, **CPU only**: Python 3.10.2, TensorFlow 2.21.0, Keras 3.12.4, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, hyperopt 0.3.0 (full list in `requirements.txt`).

## Data
`data/<name>/<name>.csv` for helpdesk, BPIC2012_A, BPIC2012_O, BPIC2012_W: the CSV logs shipped with the MTLFormer repository (the main run). They carry no Resource column, so one constant role is used. The shipped BPIC2012 files differ from the public XES logs. `camargo_inputs/*_with_resource.csv` are logs rebuilt from the XES files with a discovered resource role (a variant). See `data/DATA.md`.

## Setup
1. Clone this repository.
2. Create an environment and install the requirements:
   ```powershell
   python -m venv .venv ; .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```
3. Run the commands below from the repository folder.

## Run
Run all commands from the repository root (the folder containing `run_camargo.py`). Run `r` uses seed `42 + r - 1`. Finished runs are skipped, so a command can be stopped and restarted.

Main result used in the report (shipped CSVs, same test cases as STLFormer, LSTM dropout 0, 3 seeds):
```powershell
python run_camargo_all.py --inputs_dir data --suffix .csv --split stl --stl_dir data --runs 3 --seed 42 --lstm_dropout 0.0 --tag shipped_nodrop
```
Quick check on W only (one run):
```powershell
python run_camargo_all.py --inputs_dir data --suffix .csv --split stl --stl_dir data --datasets BPIC2012_W --runs 1 --seed 42 --lstm_dropout 0.0 --tag shipped_nodrop
```
Variants:
- As released (the repository's own timeline 80/20 split, XES-derived logs): `python run_camargo_all.py --inputs_dir camargo_inputs --runs 3 --seed 42`
- Repository hyperopt search instead of the fixed configuration (`--max_eval` is not specified in the paper; we used 10): `python run_camargo_all.py --inputs_dir data --suffix .csv --split stl --stl_dir data --search --max_eval 10 --runs 3 --seed 42 --lstm_dropout 0.2 --tag search`
- - Repository dropout 0.2 (collapses on W under Keras 3, see change 11; results in `results/split_stl_shipped`): `python run_camargo_all.py --inputs_dir data --suffix .csv --split stl --stl_dir data --runs 3 --seed 42 --tag shipped`
- Single dataset, one run: `python run_camargo.py --name helpdesk --csv data/helpdesk/helpdesk.csv --seed 42 --epochs 200`

Fixed configuration used (one trial): shared_cat, window 5, LSTM size 100, tanh, lognorm, Nadam 0.002, batch 32, at most 200 epochs, early stopping with patience 40.
Search space (with `--search`, identical to the original `dg_training.py`): model_type {shared_cat, concatenated}, n_size {5,10,15}, l_size {50,100}, lstm_act {selu,tanh}, norm_method {max,lognorm}; the best trial is chosen by validation loss; hyperopt is seeded with the run seed.

## Output
`results/split_<timeline|stl>[_<tag>]/seed<N>/<dataset>/{results_per_k.csv,summary.csv}`. Accuracy is next activity on prefixes with at least one real event (the same prefix set as STLFormer), reported as the mean over prefix lengths k (equal weight per k) and pooled over all predictions.

Summarise over seeds (mean and sample SD, `ddof=1`; writes `summary_camargo.csv`):
```powershell
python summarize_camargo.py
```

## Results folders
`results/` holds the per-run outputs of every variant we ran; `summary_camargo.csv` summarises all of them.

| Folder | Contents |
|---|---|
| `split_stl_shipped_nodrop` | Main result used in the report: shipped CSVs, STL split, LSTM dropout 0, seeds 42-44 |
| `split_stl_shipped` | Same, with the repository's dropout 0.2 (collapses on BPIC2012_W) |
| `split_stl_xes_nodrop` | Logs rebuilt from the public XES files (`camargo_inputs/`), dropout 0 |
| `split_stl` | XES-derived logs, dropout 0.2, the repository's original (unstable) sort |
| `split_stl_stable` | XES-derived logs, dropout 0.2, stable sort |
| `split_stl_stable_search` | Repository hyperopt search (seed 42, 10 trials), stable sort |
| `split_timeline`, `split_timeline_stable`, `split_timeline_search` | The repository's own timeline split (partial runs) |

## Expected results
Next-activity accuracy (%), shipped CSVs, STL split, dropout 0, seeds 42-44, mean ± sample SD over runs:

| | Helpdesk | BPIC2012_A | BPIC2012_O | BPIC2012_W |
|---|---|---|---|---|
| mean over k | 82.61 ± 0.65 | 74.81 ± 0.00 | 85.96 ± 0.00 | 88.56 ± 2.23 |
| pooled | 79.76 ± 0.16 | 71.82 ± 0.00 | 87.42 ± 0.00 | 88.71 ± 1.02 |

Paper (Table 4): 76 / 79 / 85 / 83. A and O have zero SD over the three seeds in the mean-over-k aggregation; we have not found the cause. Numbers can differ in the last digits on other hardware or library versions.

## Changes to the original code
1. `readers/log_reader.py`: pm4py import made lazy (not needed for CSV input).
2. `model_training/embedding_training.py`: `fit_generator` -> `fit`; `batch_size` cast to int (Keras 3).
3. `support_modules/role_discovery.py`: `scipy.array` -> `numpy.array`.
4. `model_training/features_manager.py`: weekday taken from the one-timestamp column (KeyError otherwise).
5. `model_training/model_optimizer.py`: `scale_args` converted to plain floats (numpy 2 repr broke `ast.literal_eval`).
6. `model_prediction/model_predictor.py`: `dim` derived from `n_size` if missing; the repository's evaluator call (wrong arguments) is bypassed and the metrics are computed in `run_camargo.py`.
7. `model_prediction/next_event_predictor.py`: loads the model from a path; shared_cat prediction batched (same predictions, much faster).
8. `model_training/model_trainer.py`: the optimizer receives the train-only log when `val_inside_train` is set.
9. `run_camargo.py`: seeds, one-timestamp mode, fixed configuration (`--search` for the full search, hyperopt seeded with the run seed), LSTM `implementation=2` (implementation 1 with dropout fails in Keras 3, same math), optional STL split.
10. `run_camargo.py`: every pandas `sort_values` is stable. The repository's unstable default sort shuffles events of a case that share a timestamp (48.6% of Helpdesk events, about 10-12% of A and O, 0% of W). Disable with `--legacy_sort` for the original behaviour.
11. `model_training/models/model_*.py` and `--lstm_dropout` (default 0.2 = repository): the LSTM dropout is now an option. With Keras 3 and dropout 0.2 the trained model predicts one class for every input on BPIC2012_W; with dropout 0 the same model trains normally.

## Known issues
- The repository's own timeline split leaks the test region into validation (as released); use the STL variants for a clean comparison.
- The time output is degenerate on date-granular logs; only next-activity accuracy is compared.
- Full runs early-stop at about epoch 40-45; a run takes about 5.5 h for the main run (4 logs x 3 seeds) on an 8-core CPU; per run about 6-10 min on Helpdesk, 19-23 min on O, 30-44 min on W, 32-67 min on A.

## Licence and citation
Original code: Apache 2.0, see `LICENSE` and `THIRD_PARTY.md`. Our changes are released under the same licence. Cite Camargo, Dumas, Gonzalez-Rojas, "Learning accurate LSTM models of business processes", BPM 2019.