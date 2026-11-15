# FEDLEGAL-India: three independent judicial FL task baselines

## Status / provenance (read first)

This package is a **new, self-contained FEDLEGAL-inspired PyTorch adaptation**, designed for the Indian judiciary project. It **is NOT an upstream FEDLEGAL fork with upstream trainers running in place**, and it has **NOT yet been trained on the Indian court dataset**. The original upstream repository can be preserved *unaltered* alongside this package using `bash scripts/checkout_upstream.sh`. This provenance distinction matters for publication. No released Chinese FEDLEGAL scores can be quoted as Indian-data results.

Sources:
- Original FEDLEGAL: https://github.com/SMILELab-FL/FedLegal (ACL 2023, Apache-2.0)
- Indian project: https://github.com/biswakim/Federated-Learning-for-Heterogeneous-and-Private-Indian-Judicial-Data
- Reference PDF: *Federated Learning for Heterogeneous and Private Indian Judicial Data*, Nov. 15, 2026.

The upstream project could be inspected on GitHub, but fetching it in this execution environment failed because of network restrictions. `scripts/checkout_upstream.sh` will fetch it on a machine with GitHub access; the training command below does *not* import or call those upstream trainers. If exact upstream FEDLEGAL trainer preservation is mandatory for your study, an integration patch against a locally cloned upstream checkout remains necessary.

## What is implemented?

Three **separate federated tasks**, each with its own global model and training run:

| Task | Label | Architecture modification | Loss | Primary test metric |
|---|---|---|---|---|
| Duration | `log1p(duration_days)` | Replace classification output with **one real-valued log-day regression output** | Huber(delta=1.0) | MAElog ↓, MAE-days ↓, Spearman ↑, 4-bin macro-F1 ↑ |
| Case type | 10 shared families | **10-class linear head** | Cross-entropy | Macro-F1 ↑, accuracy ↑ |
| Disposal | 4 categories: allowed, dismissed, disposed, withdrawn | **4-class linear head**; unclear labels excluded | Court-weighted cross-entropy (optional) | Macro-F1 ↑, accuracy ↑ |

**Kept as controls**: five natural court clients; full participation per round; weighted model averaging by local training sample count; one local epoch/round; 30-round cap and 4-round early stop; InLegalBERT + LoRA rank 8 on query/value; 256 token head+tail input; three seeds; identical temporal partitions and court-level test reporting.

**Deliberately omitted from baseline**: TSR, PTM, local personalization, joint multi-task heads, and disposal-from-case-type cascade. Thus this is an **independent single-task FL** control, not an architecture-matched joint-multi-task baseline. For fairness, also compare your model against separate task-specific models with a matched *total* compute/communication budget.

## How to use it

1. Copy your project's `judgments_labelled.csv` to `data/`, or pass its absolute path. This ZIP does not contain that dataset.
2. Set column names explicitly. The example defaults (`court`, `text`, `decision_date`, `duration_days`, `case_type`, `disposal`) are **NOT verified against your real CSV**. See data/SCHEMA.md and `--help`.
3. Prefer `--split-col split` **only if** your CSV actually has the exact original project's train/validation/test assignments; otherwise the runner reconstructs 70/15/15 chronological per-court splits. This may differ at split boundaries from original code. Match split IDs exactly for paper comparisons.
4. For exact masking comparability, provide already-masked texts from your original pipeline with `--text-already-masked`. Otherwise this package uses its own conservative mask and you must audit differences.

```bash
python -m pip install -r requirements.txt
# optional: keep an unmodified upstream copy, for reference and future patches
bash scripts/checkout_upstream.sh

# inspect CLI options
python -m fedlegal_india.train --help

# example: replace column names if your actual file differs
python -m fedlegal_india.train \
  --data data/judgments_labelled.csv --task duration --seed 0 \
  --court-col court --text-col text --date-col decision_date \
  --duration-col duration_days --model law-ai/InLegalBERT

python -m fedlegal_india.train --data data/judgments_labelled.csv --task case_type --type-col case_type
python -m fedlegal_india.train --data data/judgments_labelled.csv --task disposal --disposal-col disposal

# run all 3 tasks x 3 seeds (after verifying CSV schema / splits / masking)
export DATA=data/judgments_labelled.csv
bash scripts/run_all.sh --split-col split --text-already-masked
```

On Windows PowerShell, invoke each `python -m fedlegal_india.train` command directly, or run `run_all.sh` using Git Bash.

### Smoke test without downloading InLegalBERT

```bash
python -m pytest -q
python tests/make_smoke_data.py
python -m fedlegal_india.train --data data/synthetic_smoke.csv --task duration --toy --rounds 2 --patience 2 --batch-size 16
python -m fedlegal_india.train --data data/synthetic_smoke.csv --task case_type --toy --rounds 2 --batch-size 16
python -m fedlegal_india.train --data data/synthetic_smoke.csv --task disposal --toy --rounds 2 --batch-size 16
```

The `--toy` flag uses a randomly initialized tiny embedding model and is **not** a scientifically valid baseline. It checks only the data, training, aggregation, checkpoint, and evaluation flow.

## Evaluation artifacts

For each task and seed, `outputs/<task>/seed_<seed>/` includes:
- `best.pt`: best-validation global trainable weights (not a full standalone pretrained encoder checkpoint)
- `history.csv`: validation loss per round
- `per_court_test.csv`: per-court test metrics
- `summary.json`: five-court unweighted macro average of metrics and worst-court score
- `config.json`: full configuration

Run `python -m fedlegal_india.summarize outputs` to create `outputs/comparison.csv` and `outputs/all_seeds.csv`.

## Comparability cautions

- Three separate models use more trainable model instances and aggregate communication than one shared multi-task model; report total task-level compute and transferred bytes.
- The baseline currently reports global-model performance for every court, with **no court-level fine tuning**.
- Target mapping and preprocessing must match your project exactly, especially the case-type taxonomy and the operative ending exclusion. Masking cannot remove an operative ending by itself.
- The duration regression training target is `log1p(days)`. MAElog is not days.
- The original PDF's published results are from joint three-head prediction and cannot automatically be relabeled as FEDLEGAL-India results.
- The dataset has decisions from 2024 only, so the experiment is retrospective, not prospective decision support.
- No secure aggregation or differential privacy; FL alone is not a formal privacy guarantee.

## Baseline paper label

> **FEDLEGAL-inspired independent-task FL (adapted to Indian High Court judgments)**, implemented as separate document-level duration regression, case-type classification, and disposal-nature classification tasks. We preserve FEDLEGAL's comparative federated NLP philosophy, but **do not claim direct use of the original upstream trainer classes**. Five real court partitions and a common InLegalBERT-LoRA backbone are used for controlled comparison.

## Paper-identical metric reporting (new)
After all three tasks finish for the same seed:

```bash
python -m fedlegal_india.paper_report --runs outputs --out outputs/paper_report
```

Output: `paper_table.md`, `paper_table_mean_std.csv`, `paper_table_per_seed.csv`, `paper_metrics_per_seed.csv`, and `court_results.csv`.
These combine **three independently trained models** under the same metric columns and court averaging as Table 7 of the paper: MAElog, Spearman, duration four-bin F1, disposal macro-F1, case type macro-F1, worst-court MAElog, worst-court disposal F1. An additional MAE in days and worst-court case-type F1 are retained in the CSV.
The duration bucket boundaries are correctly inclusive of 30, 180, and 730 days (the earlier version treated exact boundary values incorrectly). Values across courts are unweighted means; the reported seed standard deviation uses ddof=1.
**Comparability requirements:** Use the exact same judgment IDs, cleaned labels, operative-ending removal, text masking, chronological split assignments, label families, and token selection as the original paper. Do not compare a 256-token single-view baseline to a 512-token TSR model as a controlled *architecture* ablation. These are separately trained single-task models and consume different total compute from one joint multi-task model.
**No real data was included in this ZIP** and no real-data scores are asserted.

## Evaluate existing checkpoints (no retraining)

This repository's `best.pt` stores **trainable LoRA weights and the task head**, not a fully standalone model. Keep the matching `config.json`, pretrained backbone reference, and tokenizer available. The evaluation command reconstructs the model, loads `best.pt`, and measures the test split only:

```bash
python -m fedlegal_india.evaluate_checkpoint --checkpoint outputs/duration/seed_0/best.pt --data data/judgments_labelled.csv
python -m fedlegal_india.evaluate_checkpoint --checkpoint outputs/case_type/seed_0/best.pt --data data/judgments_labelled.csv
python -m fedlegal_india.evaluate_checkpoint --checkpoint outputs/disposal/seed_0/best.pt --data data/judgments_labelled.csv
python -m fedlegal_india.paper_report --runs outputs --out outputs/paper_report
```

Repeat with `seed_1` and `seed_2` after training those models. Run on the **baseline package's own checkpoints**; checkpoints from the separate TSR/PTM research model have a different architecture and cannot be loaded directly into these baseline heads. Metrics use five-court unweighted averages, with the worst-court result and 3-seed mean/std. No test labels are used for training or model selection.
