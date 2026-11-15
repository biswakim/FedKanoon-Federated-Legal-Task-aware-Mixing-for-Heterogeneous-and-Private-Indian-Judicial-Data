#!/usr/bin/env bash
set -euo pipefail
: "${DATA:?Export DATA as path to your labeled CSV}"
EXTRA=("$@")
for seed in 0 1 2; do
 for task in duration case_type disposal; do
  python -m fedlegal_india.train --data "$DATA" --task "$task" --seed "$seed" "${EXTRA[@]}"
 done
done
python -m fedlegal_india.summarize outputs

python -m fedlegal_india.paper_report --runs outputs --out outputs/paper_report
