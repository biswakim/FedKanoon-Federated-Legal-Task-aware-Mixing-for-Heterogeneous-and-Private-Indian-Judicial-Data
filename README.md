# Federated Learning for Heterogeneous and Private Indian Judicial Data

Federated Learning Course project, IISER Bhopal (2026): Bijinepalli Surat Dhani, Poorvai Chandrasen, Palak Kumari,
Biswadeep Chakraborty.

**Full report:** [`report/main.pdf`](report/main.pdf)

## What this project does

Given the text of a court judgment, the model predicts three things:

| Task | Question | Output |
| --- | --- | --- |
| 1. Case duration | How long did the case take, from registration to decision? | number of days (predicted as log(1 + days)) |
| 2. Case type | What kind of case is it? | one of 10 families: writ, bail, criminal, civil, contempt, arbitration, review, tax, motor accident, other |
| 3. Disposal nature | How was it decided? | allowed, dismissed, disposed of, or withdrawn |

The model is trained with **federated learning (FL)**. Each High Court keeps its own judgments and
trains the model locally; only small model updates (about 1% of the model) are sent to a server,
which combines them. No judgment ever leaves its court.

We simulate a federation of five High Courts (Bombay, Delhi, Karnataka, Kerala, Madhya Pradesh)
using 12,360 public judgments decided in 2024.

## Objectives

**RQ1**: Are the courts measurably non-IID, and in which of the three
tasks?

**RQ2**: Does federated training beat what a court can do alone, how
does it compare with existing federated algorithms, and how
close does it get to centralized training on pooled data?

**RQ3**: What does federation give a court that it cannot get alone,
when its own data are scarce or when it has no labelled data
at all?

## Main results

**1. The courts are clearly different.** The same type of case takes a different time and ends
differently in each court, and each court writes judgments in its own style: a simple classifier
can tell which court wrote a judgment with 99.97% accuracy, even with names, places and numbers masked.

**2. Standard federated learning does not help, but personalised federated learning does.**
"Gain over local" below is the average improvement over each court training alone, across the
three tasks. All rows use the same model and 3 seeds.

| Method | Duration error ↓ | Case-type F1 ↑ | Disposal F1 ↑ | Gain over local | Hardest court |
| --- | --- | --- | --- | --- | --- |
| Each court trains alone (local) | 1.371 | 0.873 | 0.533 | 0.0% | 0.0% |
| FedProx (standard FL) | 1.383 | 0.775 | 0.532 | −4.1% | −9.3% |
| FedAdam (standard FL) | 1.369 | 0.792 | 0.526 | −3.4% | −6.9% |
| Ditto (personalised FL) | 1.329 | 0.872 | 0.555 | +2.4% | −0.3% |
| **Ours: TSR + PTM + fine-tuning** | 1.359 | **0.917** | 0.559 | **+3.6%** | **+0.6%** |
| Centralized (all data pooled, not private) | 1.358 | 0.860 | 0.559 | +1.5% | −3.9% |
| Centralized + per-court fine-tuning | 1.333 | 0.906 | 0.568 | +4.4% | +1.3% |

- Our method beats training alone (p = 0.015) and beats FedProx and FedAdam (p < 0.001).
- It is statistically tied with the best centralized model (p = 0.30), so federation gets the
  benefit of pooling data **without pooling it**.
- Case type gains most, because each court labels and writes case types in its own way.

**3. Federation helps small and new courts.** With only 25% of its data, a court gains 7.3% from
federation. A court with no labelled data at all gets a better model from the federation than
from any single other court in 4 of 5 cases.

**Note:** Our PTM method is tied with other personalised methods (Ditto, FedPer) in
accuracy; its added value is that it shows, per court and task, how much each court relies on
sharing. TSR's accuracy gain mostly comes from reading more text: a plain model reading 512
tokens does as well. See the report for details.

## How the model works

- **Encoder:** InLegalBERT (a BERT model trained on Indian legal text) with small LoRA adapters.
  Only the adapters and the prediction heads are trained and shared.
- **Three prediction heads**, one per task. Disposal nature also gets the predicted case type as input.
- **TSR (task-specific reading):** the first 256 and last 256 tokens of a judgment are read
  separately, and each task learns which part to rely on. Case type reads the start, disposal the
  end, duration both.
- **PTM (per-task mixing):** each court keeps a shared head and its own head for every task and
  learns how much to trust each. The court's own head never leaves the court.
- **Leakage control:** dates, years, numbers and outcome words are masked, and the final order is
  removed, so the model cannot read the answer from the text.

## Using the repository

### 1. Install

```bash
git clone https://github.com/biswakim/Federated-Learning-for-Heterogeneous-and-Private-Indian-Judicial-Data.git
cd Federated-Learning-for-Heterogeneous-and-Private-Indian-Judicial-Data
pip install -r requirements.txt
```

### 2. Add the data

The data are not in the repository. Put the labelled judgments file at
`data/judgments_labelled.csv` (columns are described in [`data/Readme.md`](data/Readme.md)).
It is built from the Indian High Court Judgments dataset (Dattam Labs, AWS Open Data, CC-BY-4.0).

### 3. Check that everything works (CPU, about a minute, no model download)

```bash
python tests/smoke_test.py --data data/judgments_labelled.csv --regime fl --algo fedavg
```

### 4. Run experiments

A GPU is needed for real runs; we used Kaggle (2 × T4). The easiest way is the Kaggle notebooks in
[`notebooks/`](notebooks/README.md):

| Notebook | What it runs |
| --- | --- |
| `kaggle_revision.ipynb` | the final experiments: baselines, centralized comparison, 512-token control, low-resource and unseen-court studies |
| `kaggle_fedlegal_baseline.ipynb` | the FedLegal-style baseline (one separate model per task) |
| `kaggle_experiments.ipynb` | earlier experiments: setting selection, TSR and PTM |
| `kaggle_continual.ipynb` | courts joining the federation one at a time |

To run single experiments from the command line:

```bash
DATA=data/judgments_labelled.csv
COMMON="--data $DATA --model law-ai/InLegalBERT --lr 5e-4 --max_len 256 --rounds 30 --patience 4"

# our method: TSR + PTM + per-court fine-tuning
python fl_judicial.py $COMMON --regime fl --algo fedavg --segments 2 --mix_heads dur,typ,disp --finetune_epochs 1

# each court alone, and all data pooled
python fl_judicial.py $COMMON --regime local   --epochs 15 --patience 3 --segments 2
python fl_judicial.py $COMMON --regime central --epochs 15 --patience 3 --segments 2

# other federated algorithms
python fl_judicial.py $COMMON --regime fl --segments 2 --algo fedprox --mu 0.01
python fl_judicial.py $COMMON --regime fl --segments 2 --algo ditto --ditto_lambda 1.0
```

Or run a whole set of experiments, with time budget and automatic resume:

```bash
python run_queue.py --data $DATA --plan revision --part baselines   # FL algorithms and centralized
python run_queue.py --data $DATA --plan revision --part controls    # 512-token control, extra seeds
python run_queue.py --data $DATA --plan revision --part flhelps     # low-resource and unseen-court
```

Each run saves its results to `runs/<run_name>/` and adds them to `runs/all_results.csv`.
Finished runs are skipped, and interrupted federated runs resume from the last round.

### 5. Make the tables and statistics

```bash
python scripts/revision_tables.py --runs runs --out results/revision            # result tables
python scripts/significance.py --all runs/all_results.csv \
    --a fedavg+ft --run_a fl_fedavg_ft1.0_sp_mix --b local_only --run_b local_sp_  # significance test
python scripts/heterogeneity_stats.py --data $DATA --out results/heterogeneity_text  # are courts different?
```

### Useful options of `fl_judicial.py`

| Option | Meaning |
| --- | --- |
| `--regime local / central / fl` | each court alone / all data pooled / federated |
| `--algo fedavg, fedprox, fedadam, scaffold, ditto, ...` | federated algorithm |
| `--segments 2` | use TSR (read start and end separately) |
| `--mix_heads dur,typ,disp` | use PTM for these tasks |
| `--finetune_epochs 1` | fine-tune on each court after federated training |
| `--court_frac bombay:0.25,...` | keep only part of a court's training data |
| `--holdout_court kerala` | leave one court out of training and only test on it |
| `--save_preds` | save per-case predictions, for significance tests |

## Repository layout

```
fl_judicial.py              main script: data, model, local / centralized / federated training
run_queue.py                runs sets of experiments with a time budget and resume
summarize_results.py        mean ± std over seeds
scripts/
  heterogeneity_stats.py    statistical tests: how different are the courts?
  significance.py           headline metric and significance tests
  revision_tables.py        result tables from runs/all_results.csv
  check_masking.py          checks what survives the masking
  linear_baseline.py        fast CPU baseline (n-grams, linear model)
  ner_features.py, plot_forgetting.py   optional extras
FedLegal_India_Three_Task_Baseline/   FedLegal-style baseline (see below)
notebooks/                  Kaggle notebooks
tests/smoke_test.py         quick CPU test
results/
  revision/                 final results: all_results.csv, tables, significance tests
  heterogeneity_text/       heterogeneity tests on the 12,360 judgments
  heterogeneity/            same tests on all 2024 metadata
  main_results.csv          results of the earlier report version
report/
  main.pdf, main.tex        final report
data/                       put judgments_labelled.csv here
```

## FedLegal-style baseline

[`FedLegal_India_Three_Task_Baseline/`](FedLegal_India_Three_Task_Baseline/README.md) trains a
separate federated model for each task (plain FedAvg, no personalisation), similar to how FedLegal
benchmarks legal tasks. It is inspired by FedLegal, not its original code. It uses the same split,
masking and metrics as the main model; `FedLegal_India_Three_Task_Baseline/scripts/make_adapter_csv.py`
converts the data. Run it with `notebooks/kaggle_fedlegal_baseline.ipynb`.

## Limitations

- The data are public judgments of cases decided in 2024. The privacy motivation is about
  confidential material (for example before a case is filed), which we could not test.
- Case-type and disposal labels are rule-based.
- GPU runs are not exactly reproducible; repeated runs differ by up to about 0.02 in duration error.

## Acknowledgements and license

Data: Indian High Court Judgments (Dattam Labs, AWS Open Data, CC-BY-4.0). Encoder: InLegalBERT
(`law-ai/InLegalBERT`); LoRA via `peft`. Code: MIT (see `LICENSE`); the dataset keeps its own license.
