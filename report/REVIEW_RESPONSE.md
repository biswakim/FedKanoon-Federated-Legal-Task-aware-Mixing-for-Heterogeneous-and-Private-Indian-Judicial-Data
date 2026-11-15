# Response to review: final results

All numbers below come from `results/revision/` and `results/heterogeneity_text/`. Unless stated
otherwise they use InLegalBERT + LoRA with the TSR input and 3 seeds. Δ_MTL is the mean relative
gain over local-only training across the three tasks. p-values are Wilcoxon signed-rank tests over
the 15 (court, seed) pairs. The report text is in `report/revision_sections.tex`.

## 1. Is the data heterogeneous? Yes, with statistics (12,360 training judgments)

| Kind | Result |
| --- | --- |
| Duration differs by court | Kruskal–Wallis H = 555, p < 1e-100, ε² = 0.045 |
| Case-type mix differs | χ²(36) = 4,045, Cramér's V = 0.29 (≈ Dirichlet α 1.6 per class) |
| Outcome mix differs | χ²(12) = 808, V = 0.15 |
| Same case type, different court → different duration | R² 0.028 → 0.127 when court is added (F = 39, p < 1e-90) |
| Same case type, different court → different outcome | median V = 0.22 within families |
| Courts write differently | court identified from masked text with 99.97% accuracy (chance: 20%) |
| A model from another court fails | duration error 1.32 → 2.01; disposal F1 0.65 → 0.25 |

The label imbalance is only moderate. The courts differ most in how the same kind of case behaves
and in how judgments are written.

## 2. Comparison with FedLegal-style algorithms and with centralized training

| Method | Δ_MTL vs local | Worst court | p vs ours |
| --- | --- | --- | --- |
| Local only | 0.0% | 0.0% | 0.015 |
| FedProx | −4.1% | −9.3% | < 0.001 |
| FedAdam | −3.4% | −6.9% | < 0.001 |
| FedPer | +2.9% | −2.0% | 0.17 |
| Ditto (λ = 1, chosen on validation) | +2.4% | −0.3% | 0.15 |
| **TSR + PTM + ft (ours)** | **+3.6%** | **+0.6%** | — |
| Centralized (IL-TUR-style, pooled data) | +1.5% | −3.9% | — |
| Centralized + per-court ft | +4.4% | +1.3% | 0.30 |

* Standard FL algorithms lose to local training. Per-task personalisation is required.
* Ours beats local training (p = 0.015) and FedProx/FedAdam (p < 0.001).
* Ours ties Ditto and FedPer (no significant difference) and ties centralized training with
  per-court fine-tuning (p = 0.30; bootstrap intervals include zero in every seed). In short,
  federation reaches centralized-level accuracy without moving any judgment.
* SCAFFOLD diverged (−37%) with the AdamW local optimizer, so it is reported only in a footnote.

## 3. Tasks: duration, case type, disposal nature (articles and retrieval dropped)

## 4. Metric: MAE_log for duration, macro-F1 for case type and disposal; Δ_MTL as the single headline number

## 5. How FL helps (one seed each)

* **Low-resource courts:** with 25% of the data, federation gives +7.3% over local training (case-type
  F1 0.63 → 0.81), about two thirds of what pooling the data gives. At 10% and 50% the gain is
  smaller (+2.3% and +1.4%), and at 10% duration gets worse than local training.
* **Unseen court:** for 4 of the 5 courts, the federated model trained on the other courts beats the
  best single other court's model and centralized training on the same four courts. Madhya Pradesh
  is the exception. All of these are 16–44% below the court's own local model, so a new court should
  join and fine-tune.

## Claims that changed

* **TSR:** a plain model reading 512 tokens matches TSR (p = 0.98). The TSR duration gain comes from
  reading more text. Keep TSR as an interpretability result: which part of the judgment each task reads.
* **PTM:** it ties Ditto and FedPer in accuracy. Its value is the stable sharing weights per court and
  task (duration 0.34–0.44, disposal 0.15–0.20, case type 0.04–0.08).
* **Do not claim** that FL beats centralized training, or that the methods beat IL-TUR. The tasks
  differ, and our "centralized" row is the IL-TUR-style approach.
