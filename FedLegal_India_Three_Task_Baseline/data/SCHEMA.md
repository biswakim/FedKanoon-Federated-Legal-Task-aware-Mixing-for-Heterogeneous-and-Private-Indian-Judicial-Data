# Dataset schema

No private or downloaded judicial CSV is shipped in this ZIP. Required columns (configurable):

- `court`: Bombay, Delhi, Karnataka, Kerala, Madhya Pradesh (or `MP`).
- `text`: ideally your project's already-masked judgment text **with operative order removed**.
- `decision_date`: ISO `YYYY-MM-DD`, used for chronological 70/15/15 split **only when no `split` column exists**.
- `duration_days`: nonnegative numeric number of days from registration to disposal; converted to `log1p` inside the runner.
- `case_type`: one of `writ`, `bail`, `criminal`, `civil`, `contempt`, `arbitration`, `review`, `tax`, `motor accident`, `other`.
- `disposal`: one of `allowed`, `dismissed`, `disposed`, `withdrawn`; `unclear` is excluded for disposal training.
- `split` (optional, strongly preferred): exactly `train`, `val`, `test` matching your paper.

If data is organized differently in your original repository, export an adapter CSV with these columns while **preserving the source split and mask**, rather than guessing new labels.

No label is imputed. Missing duration and unknown type/outcome labels are excluded *only for that task*.
