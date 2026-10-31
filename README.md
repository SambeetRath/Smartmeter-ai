# SmartMeter AI

SmartMeter AI is an energy data analytics platform built around the UCI Individual Household
Electric Power Consumption dataset. Milestones 1 and 2 provide CSV ingestion and quality
reporting, normalized Parquet output, and PostgreSQL persistence.

## Dataset source and attribution

- Dataset: [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- The dataset represents one household, not multiple physical meters.
- Review UCI's dataset terms before redistribution. Keep the original data unchanged and out of Git.
- Place the raw file at `data/raw/household_power_consumption.txt`.

The database design supports future meters, but the importer registers this data under the single
identifier `uci_household_1`.

## Architecture

- `src/smartmeter/ingestion/`: parses the UCI delimited file and writes processed Parquet.
- `src/smartmeter/validation/`: reports missing measurements, invalid data and exploratory IQR flags.
- `src/smartmeter/anomaly_detection/`: configurable statistical detectors, feature engineering,
  Isolation Forest, evaluation utilities, and a chronological end-to-end detection pipeline.
- `src/smartmeter/database/`: SQLAlchemy models, PostgreSQL connection/session management and repositories.
- `src/smartmeter/services/database_ingestion.py`: streams Parquet row batches into PostgreSQL.
- `src/smartmeter/api/`: FastAPI application foundation and database-independent health route.
- `alembic/`: versioned database schema migrations.
- `tests/unit/` and `tests/integration/`: database-independent tests and opt-in real PostgreSQL tests.

## Prerequisites and installation

Use the existing project venv or create it once. Python 3.12+ is required; the current project
environment has also been tested with Python 3.14.

```powershell
cd "C:\Users\sambe\SmartMeter AI"
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Runtime dependencies are pinned in `requirements.txt`; the editable install includes development
tools from the `dev` extra. The Parquet pipeline and database importer require PyArrow, SQLAlchemy,
Alembic, and Psycopg 3.

## Milestone 4, Phase 1: API foundation

The API currently provides only a liveness endpoint; it does not depend on PostgreSQL, and importing
the application does not open a database connection. OpenAPI is available at `/openapi.json`,
Swagger UI at `/docs`, and ReDoc at `/redoc`.

Start the local development server on loopback:

```powershell
.\.venv\Scripts\python.exe -m uvicorn smartmeter.api.main:app --host 127.0.0.1 --port 8000 --reload
```

In another PowerShell window, check health and the OpenAPI document:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/openapi.json
```

The health response is `{"status":"ok","service":"smartmeter-ai"}`. This endpoint confirms
application liveness only; it does not report database readiness.

## Milestone 1: data pipeline

Place the UCI file at `data/raw/household_power_consumption.txt`, then run:

```powershell
.\.venv\Scripts\python.exe -m smartmeter.ingestion.loader
```

The original CSV is not changed. All records and null measurement values are kept in
`data/processed/household_power_consumption.parquet`; the quality report is written to
`reports/quality_report.json`. Override the Parquet destination with
`SMARTMETER_PROCESSED_DATASET_PATH`.

The global 1.5×IQR flags are exploratory. They do not account for time-of-day or seasonal
consumption patterns and must not be interpreted as confirmed anomalies or removed data. A
time-aware detector is available in Milestone 3 as a separate past-only statistical baseline.

## Milestone 3, Phase 2: statistical anomaly detectors

The detectors are available as independent Python functions in
`smartmeter.anomaly_detection` and accept the normalized DataFrame schema, including the
`timestamp` column and the seven measurement columns. They return long-form results with one
row per evaluated measurement (or, for IQR, one row per flagged measurement). The
`is_suspicious_outlier` and `classification` fields explicitly describe exploratory flags;
`confirmed_anomaly` is always false. Input frames are not modified.

- **Global IQR** (`detect_global_iqr`): computes each selected column's global 25th/75th
  percentiles and flags values outside `Q1 - multiplier × IQR` / `Q3 + multiplier × IQR`.
  The configurable `columns` and `iqr_multiplier` default to the seven measurement columns
  and 1.5. This is a whole-dataset descriptive baseline, so it uses future distribution
  information and can flag legitimate peaks; do not use it as a causal detector or ground truth.
- **Rolling mean/std** (`detect_rolling_outliers`): compares a value to the preceding
  `window_size` observations, excluding the current row. `threshold` controls the number of
  population standard deviations from the rolling mean. `minimum_history` (default:
  `window_size`) controls how many finite prior measurements are required. Missing and infinite
  values do not contribute. A zero-variance baseline has equal upper/lower bounds at its mean.
  Windows count observations rather than elapsed minutes.
- **Time-of-day-aware** (`detect_time_of_day_outliers`): compares against prior values in the
  same minute-of-day and, by default, same day-of-week bucket. Set `group_by_day_of_week=False`
  to pool weekdays and weekends. Expanding mean/std are calculated only from earlier
  timestamps, and no flag is evaluated until `minimum_history` finite matching observations
  are available. Both time-aware detectors require valid, unique timestamps; duplicate times
  are rejected because their temporal ordering is ambiguous.

Example invocation from PowerShell:

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; from smartmeter.anomaly_detection import detect_rolling_outliers; data = pd.read_parquet('data/processed/household_power_consumption.parquet'); flags = detect_rolling_outliers(data, columns=['global_active_power'], window_size=60, minimum_history=30, threshold=3); print(flags.loc[flags['is_suspicious_outlier']].head())"
```

Unit tests use small generated time-series data and require neither the real UCI dataset nor
PostgreSQL:

```powershell
.\.venv\Scripts\python.exe -m pytest -v tests/unit/test_statistical_anomaly.py
```

These are statistical baselines only. They are not labeled ground truth, do not establish
real-world detection accuracy, and do not yet include machine-learning evaluation.

## Milestone 3, Phase 3: feature engineering and synthetic test data

`smartmeter.anomaly_detection.features.engineer_features` adds temporal features and past-only
measurement history features to a copy of the input DataFrame. It keeps all original columns,
the original index, and source row order; it does not drop rows. The default measurement set
includes the seven normalized UCI measurements.

- Temporal features include hour, minute-of-day, weekday, weekend, month, and cyclical
  sine/cosine encodings for hour, weekday, and month.
- Lag features default to the prior 1, 5, 15, and 60 observations. Lag values are based on
  row counts, so the 60-observation lag is not guaranteed to mean exactly 60 elapsed minutes
  when the source has gaps.
- Rolling mean, population standard deviation, minimum, maximum, and deviation from the
  historical mean default to a 60-observation window. Configure `rolling_windows` to request
  additional windows. Calculations shift the series by one observation before rolling, so
  the current measurement never contributes to its own baseline.
- Source measurement columns are unchanged. Missing and non-finite values are unavailable
  for derived history calculations, and a
  `<measurement>__is_missing` feature marks them. No rows are automatically removed.
- Derived numeric features use float32 to reduce memory. The default calculates history
  features for every available measurement; pass `columns=["global_active_power"]` to reduce
  feature count and memory when only active power is needed. A large number of columns over
  the full two-million-row dataset can still require substantial memory.

Generate features from the full time-ordered source before splitting. Since each lag and rolling
statistic uses only earlier rows, validation/test features can use previously observed training
history without seeing future values. The chronological split helper sorts by timestamp and
returns disjoint train, validation, and test copies. It rejects invalid or duplicate timestamps
and fractions that yield an empty partition. Fit any later preprocessing/model only on the
training partition. Inject synthetic anomalies only into a copy of held-out validation/test data,
never into training by default.

`smartmeter.anomaly_detection.synthetic_data.inject_synthetic_anomalies` returns original
measurements, `original_<measurement>` values, bounded `synthetic_<measurement>` values, and
explicit `synthetic_anomaly_label`, `synthetic_anomaly_category`, and per-row
`synthetic_injection_metadata` fields. The seeded configuration supports:

- Sudden active-power spikes, capped at a configurable maximum.
- Unusually low active power, with a configurable nonnegative floor.
- Sustained high or low active power in non-overlapping consecutive-observation segments.
- Positive or negative voltage deviations clipped to configurable plausible bounds.

Rates are fractions of rows rounded to the nearest observation. The sustained rate is rounded
down to complete fixed-length segments. Injection selections do not overlap; infeasible
requests raise `ValueError` rather than silently reducing the requested injection count.
The default 15 kW active-power cap and 180–260 V voltage bounds are controlled test scenario
limits, not claims about universal electrical limits. Original input columns are never modified.
Synthetic labels are generated solely for controlled tests and are not verified real-world labels
or evidence of real-world anomaly-detection accuracy.

Run the Phase 3 unit tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -v tests\unit\test_feature_engineering.py tests\unit\test_synthetic_data.py
```

These utilities use the existing Pandas/NumPy dependencies and do not require the real dataset,
PostgreSQL, or a machine-learning library.

## Milestone 3, Phase 4: Isolation Forest

`smartmeter.anomaly_detection.isolation_forest.IsolationForestDetector` fits an unsupervised
scikit-learn Isolation Forest. Pass only the chronological training partition to `fit`; the
method accepts no target argument and rejects label/target columns. It stores the fitted feature
selection and median imputer, so validation and test observations cannot affect preprocessing.
Missing and non-finite numeric values are imputed using medians calculated on training rows;
features that are entirely missing in training are retained and imputed as zero.

Set `IsolationForestConfig.feature_columns` to an explicit ordered tuple to control the input
features. If omitted, numeric and boolean columns are selected automatically, excluding timestamp
fields, label-like columns, and `original_` / `synthetic_` columns. Predictions are returned as
a copy of the input with `isolation_forest_prediction` (`-1` or `1`),
`isolation_forest_anomaly_label` (`1` for the estimator's outlier decision, else `0`), and
`isolation_forest_anomaly_score` (larger values are more abnormal). The binary output is an
unsupervised model decision, not a verified anomaly label. `contamination` controls the
decision threshold; scores are not calibrated probabilities.

Create past-only features before the chronological split using the Phase 3 feature component,
then fit on training rows and score later periods. The detector itself does not model temporal
order; results depend on the supplied features and the Isolation Forest parameters.

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; from smartmeter.anomaly_detection import IsolationForestDetector, engineer_features, chronological_train_validation_test_split; data = pd.read_parquet('data/processed/household_power_consumption.parquet'); features = engineer_features(data, columns=['global_active_power']); split = chronological_train_validation_test_split(features); detector = IsolationForestDetector().fit(split.train); scored = detector.predict(split.test); print(scored[['timestamp', 'isolation_forest_anomaly_label', 'isolation_forest_anomaly_score']].head())"
```

Run the Phase 4 tests independently with:

```powershell
.\.venv\Scripts\python.exe -m pytest -v tests\unit\test_isolation_forest.py
```

Fitted detectors can be saved atomically with `save(path)` and loaded with
`IsolationForestDetector.load(path)`. Joblib artifacts use pickle serialization: load only
artifacts from trusted sources.

## Milestone 3, Phase 5: model evaluation and comparison

`smartmeter.anomaly_detection.evaluation.run_anomaly_evaluation` compares global IQR,
past-only rolling mean/std, time-of-day, and Isolation Forest using a chronological
train/validation/test split. Feature engineering runs once on the original series and uses
past observations only. Synthetic anomalies are injected into copies of validation and test
partitions; original input rows remain unchanged.

The Isolation Forest imputer and estimator fit on training rows only. Optional contamination
candidates are fit independently on the training partition and reported on validation data
only. No candidate is selected automatically, and the test evaluation uses the explicitly
configured Isolation Forest. Ground-truth labels are injected after splitting and cannot enter
training features.

The existing global IQR function remains a whole-frame exploratory detector. For evaluation,
`GlobalIQRDetector` applies the same quartile/IQR threshold rule but learns its thresholds on
training data and applies them unchanged to held-out rows. This prevents evaluation rows from
setting the thresholds; the global baseline still does not account for time-of-day or seasonal
patterns. Rolling and time-of-day detectors use historical rows, so their warm-up periods may
exclude observations. Each detector reports its available-row coverage and exclusion reasons.
Unavailable predictions are excluded from that detector's metrics, never counted as normal.
A separate shared-row summary compares all detectors on their common available test rows.

Reports include confusion matrices in `[normal=0, anomaly=1]` order, TP/TN/FP/FN, precision,
recall, F1, accuracy, false-positive rate, specificity, coverage and category-level metrics.
Undefined zero-denominator metrics are reported as `0.0`; when a detector has no available
observations, its metrics are `null`. Category analysis is one-vs-rest: other synthetic anomaly
categories are separately counted non-target negatives and are not described as normal.

Synthetic categories are controlled test scenarios, not verified real-world labels. Results
measure agreement with injected patterns only and do not establish real-world model performance.
No performance figure is included in this documentation; run the evaluation to obtain results
for the selected dataset and configuration.

Example PowerShell invocation; the default report writer refuses to overwrite an existing file.
Set `overwrite=True` only when intentionally replacing an existing report:

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; from smartmeter.anomaly_detection import EvaluationConfig, IsolationForestConfig, run_anomaly_evaluation, write_evaluation_report; data = pd.read_parquet('data/processed/household_power_consumption.parquet'); config = EvaluationConfig(dataset_identifier='uci_household_1', contamination_candidates=(0.01, 0.02, 0.05, 0.10), isolation_forest_config=IsolationForestConfig(contamination=0.05)); report = run_anomaly_evaluation(data, config=config); write_evaluation_report(report, 'reports/anomaly_evaluation.json'); print(report['test']['shared_comparison'])"
```

Run the Phase 5 unit tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -v tests\unit\test_evaluation.py
```

## Milestone 3, Phases 6–7: end-to-end detection pipeline

The pipeline is available through `smartmeter.anomaly_detection.pipeline` and the package CLI.
It reads the normalized Parquet dataset by default, or ordered readings for a configured meter
from PostgreSQL. The database source requires the existing `DATABASE_URL` configuration and an
available database. Both readers validate the timestamp and seven-measurement schema, reject
duplicate/invalid timestamps and non-numeric measurement values, retain missing values, and sort
the data chronologically without dropping records.

Run all four detectors against the processed Parquet file:

```powershell
.\.venv\Scripts\python.exe -m smartmeter.anomaly_detection `
  --source parquet `
  --input data\processed\household_power_consumption.parquet `
  --results reports\anomaly_detections.csv `
  --summary reports\anomaly_run.json `
  --detectors global_iqr rolling_mean_std time_of_day isolation_forest `
  --measurement-columns global_active_power `
  --train-fraction 0.7
```

Outputs are protected from accidental replacement. For an intentional rerun, add `--overwrite`.
The JSON summary records the run ID, source, time boundaries, input/training/held-out row counts,
detector configurations, prediction coverage, unavailable reasons, alert counts, output path and
CSV size. The CSV has one row per detector and timestamp for an alert or an unavailable
prediction. It contains the timestamp, meter identifier, original seven measurement values,
detector name, nullable prediction, availability, score, optional category, result kind, and
detector metadata. Each row's metadata identifies the corresponding detector configuration and
score semantics in the JSON summary by detector name, avoiding repeated large configuration
payloads. Available normal predictions are omitted from this sparse CSV and are counted in the
JSON summary; an unavailable prediction is never emitted as normal. Categories remain empty
because these detectors do not infer a supported event category.

The pipeline makes one chronological training/held-out split (70%/30% by default). Global IQR
thresholds and Isolation Forest are fit on training rows only. Rolling and time-of-day statistics
are causal and may use observations strictly before each held-out timestamp, including training
history. Isolation Forest features are engineered with past-only lags and rolling statistics;
synthetic labels are rejected and are never used in inference. The default Isolation Forest has a
fixed random seed. Prediction scores are detector-specific (IQR fence distance, historical
standard deviations, or negative Isolation Forest score) and are not directly comparable or
calibrated probabilities. Statistical alerts are exploratory suspicious-outlier flags, and model
alerts are unsupervised decisions; none are verified real-world anomaly ground truth.

The CLI supports individual detector selection and thresholds, for example:

```powershell
.\.venv\Scripts\python.exe -m smartmeter.anomaly_detection `
  --detectors rolling_mean_std time_of_day `
  --measurement-columns global_active_power voltage `
  --rolling-window-size 120 `
  --rolling-minimum-history 60 `
  --rolling-threshold 3.5 `
  --time-of-day-minimum-history 6 `
  --time-of-day-threshold 3.5 `
  --results reports\statistical_alerts.csv `
  --summary reports\statistical_alerts.json
```

`PipelineConfig` also accepts an `IsolationForestConfig` to control model features and estimator
parameters. To read PostgreSQL, first configure `DATABASE_URL` as in the database section, then
pass `--source postgresql --meter-identifier uci_household_1`. Database readings are fetched in
bounded pages but currently materialized in memory to support historical baselines and feature
engineering. Full-data processing was exercised against the existing 2,075,259-row Parquet file;
peak process memory was not measured.

Run the pipeline tests without the real dataset or PostgreSQL:

```powershell
.\.venv\Scripts\python.exe -m pytest -v tests\unit\test_pipeline.py
```

## PostgreSQL architecture and schema

PostgreSQL 17 runs as the `postgres` Docker Compose service. Its storage is persisted in the
named `postgres_data` volume and its port is bound to loopback for local development. The optional
`ingest` service uses the same Compose network and waits for the database health check.

The initial Alembic migration creates:

- `meters`: unique `meter_identifier`, optional `location`, and timezone-aware `created_at`.
- `meter_readings`: required meter reference and timezone-aware timestamp; seven nullable
  `NUMERIC(12, 6)` measurement fields.

The `(meter_id, timestamp)` unique constraint prevents duplicate meter/time readings and supplies
the composite lookup index. Deleting a meter with readings is restricted. UCI timestamps have no
timezone; this project interprets them as UTC at database ingestion. Missing measurements remain
SQL `NULL`.

Batch writes use PostgreSQL `ON CONFLICT DO NOTHING`, so a rerun is idempotent. Each batch commits
in its own transaction; a failed batch rolls back while previously committed batches can safely
be retried. Global IQR flags are not used to reject records. Rows with invalid timestamps,
negative measurements, or non-finite numeric values are counted as rejected; missing measurements
are accepted.

## Docker Desktop and environment

1. Start Docker Desktop and wait for its engine to become ready.
2. Create the local environment file and replace the placeholder password:

```powershell
Copy-Item .env.example .env
notepad .env
```

`.env` is ignored by Git. `.env.example` contains only local placeholders, not production
credentials. Keep `DATABASE_URL` consistent with the Postgres values in `.env`.

Start PostgreSQL and check its health:

```powershell
docker compose up -d postgres
docker compose ps
docker compose exec postgres pg_isready -U smartmeter_user -d smartmeter_db
```

The example health command assumes the values shown in `.env.example`; use the values from your
own `.env` if changed.

## Migrations and ingestion

The host-side migration and ingestion commands use `DATABASE_URL`. Set it in the current
PowerShell session to the local database URL from `.env` (URL-encode special characters in the
password):

```powershell
$env:DATABASE_URL = "postgresql+psycopg://smartmeter_user:YOUR_URL_ENCODED_PASSWORD@localhost:5432/smartmeter_db"
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m smartmeter.services.database_ingestion
```

For Docker-based ingestion, after migration has been applied:

```powershell
docker compose --profile ingestion run --rm ingest
```

The importer reads `data/processed/household_power_consumption.parquet` in batches. Configure
`SMARTMETER_INGESTION_BATCH_SIZE` (default 5000) and `SMARTMETER_METER_IDENTIFIER` (default
`uci_household_1`). The importer logs records read, inserted, skipped, and rejected.

## Tests and lint

Unit tests do not require the real 133 MB UCI CSV or a database:

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not integration" -v
.\.venv\Scripts\python.exe -m ruff check src tests
```

Real PostgreSQL integration tests require a dedicated database whose name ends in `_test`. They
run the Alembic migration and repository checks; they never drop or truncate tables.

```powershell
$env:SMARTMETER_TEST_DATABASE_URL = "postgresql+psycopg://smartmeter_user:YOUR_URL_ENCODED_PASSWORD@localhost:5432/smartmeter_test"
.\.venv\Scripts\python.exe -m pytest -m integration -v
```

To stop containers without deleting the persistent database volume:

```powershell
docker compose down
```

Do not use `docker compose down -v` unless you explicitly intend to delete database data.

## Troubleshooting

- **Docker daemon unavailable:** open Docker Desktop and wait until `docker info` succeeds.
- **Port 5432 already in use:** set `POSTGRES_PORT` in `.env` to an unused local port and update
  the host `DATABASE_URL` to match.
- **Connection refused:** run `docker compose ps` and wait for the `postgres` health check.
- **Authentication failed:** ensure `DATABASE_URL` credentials match `POSTGRES_USER` and
  `POSTGRES_PASSWORD`. If changing credentials for an existing volume, note that the existing
  Postgres data directory retains its original initialized user/password.
- **Parquet file missing:** run the Milestone 1 pipeline first, or confirm the file exists at the
  configured `SMARTMETER_PARQUET_PATH`.
- **Migration command cannot connect:** set `DATABASE_URL` in the same PowerShell session before
  running Alembic.

## Current scope

Milestones 1 and 2 cover ingestion, quality reporting, Parquet output, PostgreSQL persistence,
migrations, and batch import. Milestone 3 includes statistical detectors, past-only feature
engineering, controlled synthetic evaluation data, chronological evaluation, Isolation Forest,
and the end-to-end detection pipeline. PostgreSQL pipeline input is supported but requires a
configured database. FastAPI, dashboard, LLM integration, and complete Docker/CI deployment remain
future milestones.
