#!/usr/bin/env bash
set -euo pipefail
# Original upstream source is kept UNMODIFIED for provenance and future native integration.
# This project has its own runner. Do NOT claim it executes upstream trainers.
if [[ ! -d upstream/FedLegal/.git ]]; then
  mkdir -p upstream
  git clone https://github.com/SMILELab-FL/FedLegal.git upstream/FedLegal
fi
(cd upstream/FedLegal && git rev-parse HEAD) | tee upstream/FEDLEGAL_COMMIT.txt
echo 'FedLegal upstream preserved at upstream/FedLegal (unmodified)'
