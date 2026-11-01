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
- `src/smartmeter/services/detection_service.py` and `detection_worker.py`: run and persist
  durable detector jobs outside API request processes.
- `src/smartmeter/api/`: FastAPI health, readings, detection-run, and anomaly endpoints.
- `src/smartmeter_dashboard/`: separate Streamlit shell, typed HTTPX API client, configuration,
  and Pandas response transformations. The dashboard communicates with FastAPI only and never
  loads database credentials.
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

## Milestone 4: API and persistent detection

The API provides a database-independent liveness endpoint, bounded readings queries, durable
detection-run submission and status, and paginated persisted predictions. Importing the application
does not open a database connection. OpenAPI is available at `/openapi.json`, Swagger UI at `/docs`,
and ReDoc at `/redoc`.

Start the API from the project root or another working directory with:

```powershell
.\run_api.ps1
```

The launcher reads `DATABASE_URL` from `.env` without displaying it, uses the project virtual
environment, and runs without automatic reload. Stop any previous API instance before starting
another one on port 8000. `/health` remains database-independent; `/readings` requires the
configured, migrated PostgreSQL database.
The API launcher binds to `127.0.0.1`; authentication is not implemented, so do not expose this
development API to an untrusted network.

The local API documentation and health URLs are:

```powershell
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
http://127.0.0.1:8000/openapi.json
http://127.0.0.1:8000/health
```

The health response is `{"status":"ok","service":"smartmeter-ai"}`. This endpoint confirms
application liveness only; it does not report database readiness.

For example, query readings from PowerShell with:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/readings?meter_identifier=uci_household_1&limit=10"
```

`GET /readings` requires `meter_identifier`. Optional `start` and `end` parameters must be
ISO-8601 timestamps that include a UTC offset; they are normalized to UTC and define the
half-open interval `[start, end)`. Results are ordered by timestamp and limited to at most 1,000
per page. Responses preserve missing measurements as JSON `null` and include `has_more` and
`next_offset` pagination fields. Reading requests require a configured, migrated PostgreSQL
database; an unknown meter returns 404.

Apply the additive detection schema migration before using detection endpoints. Run the migration
from the project root after loading `DATABASE_URL` from `.env` into the current PowerShell session:

```powershell
$databaseUrlLine = Get-Content .env | Where-Object { $_ -match '^\s*DATABASE_URL\s*=' } | Select-Object -First 1
if (-not $databaseUrlLine) { throw "DATABASE_URL is missing from .env" }
$env:DATABASE_URL = (($databaseUrlLine -split '=', 2)[1]).Trim().Trim('"').Trim("'")
.\.venv\Scripts\python.exe -m alembic upgrade head
```

The detection API rejects requested ranges longer than 90 days and any selection containing more
than 100,000 readings. The worker independently enforces the reading-count limit before loading
data, including for jobs not submitted through the API. At most one run per meter may be pending
or running; competing submissions receive HTTP 409. Reading, detection-run, and anomaly pages
allow at most 1,000 items and offsets up to 100,000. These safeguards bound one job and one query;
they are not authentication, per-user quotas, or global request-rate limits.

The detection migration is present at revision `20261031_0002` and is applied to the isolated
`smartmeter_integration_test` database used for PostgreSQL verification and to the approved
development database `smartmeter_db`. Both are at revision `20261031_0002`; `alembic check`
reported no schema drift. The approved upgrade added the `detection_runs` and `anomaly_results`
tables and their indexes/constraints; it did not drop or reset existing tables or readings.

Verification on 1 November 2026: the full suite reported 180 passed and 10 PostgreSQL tests
skipped without a test URL; the separate isolated PostgreSQL suite reported 10 passed. The live
localhost API/worker verification against the isolated database returned HTTP 202 for a queued
run, HTTP 409 for a duplicate active submission, then completed the run and persisted reconciled
counts (1 anomaly, 4 normal, 1 unavailable) from 20 synthetic readings. `alembic current` reported
`20261031_0002 (head)` and `alembic check` found no drift. Ruff lint/format, dependency checks,
`check_project.ps1` (5/5), and Git whitespace checks passed. After owner approval, the development
database was migrated and verified at `20261031_0002`; live localhost checks returned 200 for
health, docs, OpenAPI, readings (2 records), detection-run listing, and anomaly listing.

Submit a durable detection job. `start` is inclusive, `end` is exclusive, and supplied timestamps
must include an ISO-8601 UTC offset:

```powershell
$body = @{
  meter_identifier = "uci_household_1"
  algorithm = "global_iqr"
  start = "2006-12-16T00:00:00Z"
  end = "2006-12-17T00:00:00Z"
} | ConvertTo-Json
$run = Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/detection/runs" `
  -ContentType "application/json" -Body $body
$run
```

The API records a `pending` run and returns HTTP 202. It does not execute detector work in the
request process. Start a worker in another PowerShell terminal after loading the same database URL
from `.env`:

```powershell
$line = Get-Content .env | Where-Object { $_ -match '^\s*DATABASE_URL\s*=' } | Select-Object -First 1
if (-not $line) { throw "DATABASE_URL is missing from .env" }
$env:DATABASE_URL = (($line -split '=', 2)[1]).Trim().Trim('"').Trim("'")
.\.venv\Scripts\python.exe -m smartmeter.services.detection_worker
```

The worker claims jobs with PostgreSQL row locks and expiring leases, renews leases while processing,
and retries jobs after worker crashes up to three claims. Permanently failed jobs expose a safe
failure message, not internal exception details. `--once` processes at most one available job.
Optional `SMARTMETER_DETECTION_POLL_SECONDS`, `SMARTMETER_DETECTION_LEASE_SECONDS`, and
`SMARTMETER_DETECTION_HEARTBEAT_SECONDS` settings tune worker polling and leases.

Check a run and its persisted predictions with:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/detection/runs/$($run.run_id)"
Invoke-RestMethod "http://127.0.0.1:8000/anomalies?run_id=$($run.run_id)&limit=100"
Invoke-RestMethod "http://127.0.0.1:8000/anomalies/summary?run_id=$($run.run_id)"
```

`GET /anomalies` defaults to anomalous predictions; request `prediction_status=normal` or
`prediction_status=unavailable` to inspect those categories. The summary describes exactly one run.
If `run_id` is omitted, it selects the latest completed run, optionally filtered by meter, so
repeated detection runs do not double-count readings. Run counters include training and held-out
readings separately; only chronological held-out predictions are persisted. The detection pipeline
currently materializes the selected range in memory, so each run is intentionally capped at
100,000 readings. Use a smaller date range where possible.

For containerized worker execution, start PostgreSQL and the detection profile (after applying
migrations):

```powershell
docker compose --profile detection up -d postgres detection-worker
docker compose logs -f detection-worker
```

## Milestone 5 Phase 1: Analytics API

The read-only analytics endpoints provide SQL-aggregated dashboard data through the existing
FastAPI service. Dashboard clients use the API and do not receive database credentials.

- `GET /meters?limit=100&offset=0` returns meter identifiers, stored reading counts, and first
  and last stored timestamps. It is paginated with a maximum page size of 1,000.
- `GET /analytics/summary?meter_identifier=uci_household_1&start=...&end=...` returns the
  reading count; active-power average, minimum, maximum, and latest non-null value/time; and
  missing counts and rates for all seven measurement fields. `start` and `end` are optional
  UTC-offset ISO-8601 boundaries using half-open `[start, end)` semantics.
- `GET /analytics/timeseries?meter_identifier=uci_household_1&start=...&end=...&interval=hour`
  returns PostgreSQL hourly (`hour`) or daily (`day`) bucket aggregates for all seven electrical
  measurements. Both UTC-offset boundaries are required; the maximum range is 31 days and the
  result is capped at 750 buckets.
- `GET /analytics/quality?meter_identifier=uci_household_1&start=...&end=...` returns
  database-observed missing counts/rates and internal timestamp-gap counts. Gap estimates assume
  a one-minute expected cadence; gaps before the first or after the last in-scope reading are
  not assessed. Ingestion-time validation findings are explicitly unavailable because they are
  not stored per reading and cannot be reconstructed.

Examples from PowerShell after starting the API:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/meters?limit=100"
Invoke-RestMethod "http://127.0.0.1:8000/analytics/summary?meter_identifier=uci_household_1&start=2006-12-16T00:00:00Z&end=2006-12-17T00:00:00Z"
Invoke-RestMethod "http://127.0.0.1:8000/analytics/timeseries?meter_identifier=uci_household_1&start=2006-12-16T00:00:00Z&end=2006-12-17T00:00:00Z&interval=hour"
Invoke-RestMethod "http://127.0.0.1:8000/analytics/quality?meter_identifier=uci_household_1&start=2006-12-16T00:00:00Z&end=2006-12-17T00:00:00Z"
```

Verification on 1 November 2026: 15 focused analytics unit tests passed. The full suite reported
195 passed and 11 PostgreSQL tests skipped without a test URL; the separate isolated
`smartmeter_integration_test` run passed all 11 PostgreSQL tests. Ruff lint/format, `pip check`,
Alembic drift check, and `git diff --check` passed. No database schema migration was needed.
This completes the backend analytics API phase. The separate dashboard foundation is described
below; the six analytics pages remain unimplemented.

## Milestone 5 Phase 2: Dashboard foundation

The separate `src/smartmeter_dashboard/` package contains dashboard configuration, a typed HTTPX
client for health, meter coverage, analytics, detection runs, and anomalies, Pandas transformations,
and a minimal Streamlit shell. The shell checks API availability and loads a meter selector; it
provides navigation placeholders for the six dashboard sections, but does not yet implement their
analytics content or charts.

Install the optional dashboard dependencies into the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dashboard]"
```

In PowerShell, configure the API URL only if it differs from the local default, then run
`streamlit` from the project venv in a separate terminal after starting the API with `run_api.ps1`:

```powershell
$env:SMARTMETER_API_BASE_URL = "http://127.0.0.1:8000"
.\.venv\Scripts\python.exe -m streamlit run src\smartmeter_dashboard\app.py
```

`SMARTMETER_API_BASE_URL` defaults to `http://127.0.0.1:8000`. It must be an HTTP(S) base URL
without embedded credentials, query parameters, or fragments. The dashboard does not read
`DATABASE_URL`, open a PostgreSQL connection, or provide a database credential field. Dashboard
tests use mocked HTTP responses and do not require a live API or PostgreSQL.

Phase 2 verification on 1 November 2026: 21 dashboard-focused tests passed; the full suite
reported 216 passed and 11 PostgreSQL tests skipped without a test URL. Ruff lint/format,
`pip check`, and `git diff --check` passed. The optional dashboard extra installed Streamlit
1.66.1 and Plotly 6.9.0, and reused HTTPX 0.28.1 in `.venv`.

## Milestone 5 Phase 3: navigation and shared filters

The dashboard now has sidebar navigation for Overview, Energy Analytics, Electrical Metrics,
Anomaly Detection, Detection Runs, and Data Quality. Each section is a clearly identified
placeholder; analytics content and charts are intentionally not implemented in this phase.

Meter selection and the inclusive UTC date range are shared through Streamlit session state.
The default is up to the latest seven days in the selected meter's actual coverage, with a
seven-day fallback when coverage is unavailable. Date ranges are validated and capped at 31 days.
The meter list is cached for five minutes to avoid repeat API calls during navigation; the visible
Refresh data button clears that cache and reloads it. Date boundaries exposed for API use are
UTC-midnight timestamps with an exclusive end.

Reusable components provide the application header, API status, loading, empty, and error states.
The dashboard continues to communicate only with FastAPI; it has no database connection or
database credential access.

Phase 3 verification on 1 November 2026: 37 dashboard-focused tests passed; the full suite
reported 232 passed and 11 PostgreSQL integration tests skipped because no isolated test URL was
configured. Ruff lint/format, `pip check`, and `git diff --check` passed. Tests used mocked API
responses; no live API or database was required.

## Milestone 5 Phase 4: core analytics pages

Overview, Energy Analytics, and Electrical Metrics now use the shared meter and UTC date filters
and fetch only the existing server-aggregated analytics endpoints. Successful summary, time-series,
and per-meter anomaly-summary GET responses are cached for 60 seconds; failed responses are not
cached. The Refresh data action clears these caches. Hourly buckets are bounded by the existing
31-day request window and 750-bucket API response limit; no raw readings are downloaded.

Overview displays reading-count, active-power average/maximum/latest KPIs, the API's missing-value
counts and rates, a daily active-power/reading-volume chart, and (when available) the meter's
latest completed detection-run summary. That anomaly summary is run-scoped and is not filtered by
the selected date range. Energy Analytics displays selected-period average and peak power plus
hourly and daily active-power and reading-count charts. Electrical Metrics provides hourly/daily
trends for active and reactive power (kW), voltage (V), current intensity (A), and the three
sub-metering measures (Wh). Null aggregate values remain chart gaps and are counted, not filled.

The Anomaly Detection, Detection Runs, and Data Quality sections remain placeholders for Phase 5.

Phase 4 verification on 1 November 2026: 47 dashboard-focused tests passed; the full suite
reported 242 passed and 11 PostgreSQL integration tests skipped because no isolated test URL was
configured. Ruff lint/format and `git diff --check` passed; `pip check` reported no broken
requirements. Tests use mocked API responses. A live API check at
`http://127.0.0.1:8000/health` could not connect, so live backend verification and screenshots
were not available.

## Milestone 5 Phase 5: detection and data-quality pages

The six-section dashboard now includes API-backed Anomaly Detection, Detection Runs, and Data
Quality pages. Anomaly results support shared meter/date filters, prediction-status/algorithm/
category filtering, and bounded pagination; unavailable predictions are labeled separately from
normal predictions. Detection Runs displays filtered run history and processing counts, including
unavailable predictions. Data Quality displays database-computed missing-value statistics and
timestamp-gap estimates, and explicitly reports when historical ingestion-time validation is not
available.

Shared KPI cards wrap long labels and values, data tables allocate more width to timestamps and
identifiers, and status/chart helpers preserve the existing dark theme. Dense hourly charts reduce
point markers and limit date ticks. Power metrics are labeled in kW; sub-metering is labeled as
average Wh per source reading, not kWh. Missing aggregate values remain gaps, and unavailable
predictions are never presented as normal. Dashboard requests continue to use FastAPI; no database
access was added to the dashboard.

The Overview treats only the API's explicit `No completed detection run found.` response as an
expected empty state and caches that result for 60 seconds. Unknown-meter 404s and other API
failures remain visible as errors. Refresh clears this summary cache so a newly completed run can
appear immediately.

Verification on 1 November 2026: 60 dashboard-focused tests passed; the full suite reported 255
passed and 11 PostgreSQL integration tests skipped because no isolated test URL was configured.
Ruff lint and format checks, `pip check`, and `git diff --check` passed. A live read-only API health
check returned HTTP 200, and the dashboard returned HTTP 200. Browser verification against the
running API navigated all six pages and confirmed the selected meter/date filters persisted. The
live data showed 9,903 readings for the selected 20–26 November 2010 range; no detection-run
summary was available for that meter. No database or `.env` changes were made.

Anomaly-summary follow-up verification: focused dashboard/API tests passed (52 passed); the full
suite reported 260 passed and 11 PostgreSQL integration tests skipped. Ruff lint/format, `pip
check`, and `git diff --check` passed. The local API was unavailable during this follow-up, so
the current database's summary response was not rechecked live.

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

## Milestone 6.1: assistant configuration and analytics tool registry

The `smartmeter.assistant` package provides opt-in configuration, provider-neutral Pydantic
contracts, independent OpenAI and Ollama adapters, a fixed registry of read-only analytics tool
definitions, and trusted bounded execution for those tools. Conversation orchestration, an
assistant API route, and chat UI are not implemented.

Provider selection is explicit. With `LLM_PROVIDER` unset, assistant functionality is unavailable;
FastAPI startup does not load assistant configuration. Set `LLM_PROVIDER=openai` with
`OPENAI_API_KEY` and `OPENAI_MODEL`, or set `LLM_PROVIDER=ollama` with `OLLAMA_MODEL` and an
optional `OLLAMA_BASE_URL` (defaults to `http://127.0.0.1:11434`). There is no automatic provider
fallback. `LLM_REQUEST_TIMEOUT` defaults to 30 seconds and must be between 1 and 120 seconds. API keys are
held as Pydantic `SecretStr` values.

The shared provider contract and adapters are synchronous to match the current synchronous
FastAPI route/service pattern; synchronous route handlers run in FastAPI's worker pool. The
interface contains no provider SDK types. Install only the provider you intend to use, or both:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[assistant-openai]"
.\.venv\Scripts\python.exe -m pip install -e ".[assistant-ollama]"
.\.venv\Scripts\python.exe -m pip install -e ".[assistant-openai,assistant-ollama]"
```

The `assistant-openai` extra uses the official OpenAI Python SDK and its Responses API. The
`assistant-ollama` extra declares HTTPX explicitly and uses Ollama's `/api/chat` endpoint. Imports
of either optional client library are lazy; importing the shared assistant package does not
require either provider dependency. `create_provider()` selects only the configured provider and
raises a safe actionable error if configuration or that provider's extra is missing. It never
falls back to another provider.

OpenAI uses `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, and `OPENAI_MODEL`. Ollama uses
`LLM_PROVIDER=ollama`, `OLLAMA_MODEL`, and optionally `OLLAMA_BASE_URL` (default
`http://127.0.0.1:11434`). A configured Ollama URL may be remote; the application does not start
Ollama, download models, or probe model tool-calling support in advance. If a model/server rejects
tool definitions, the adapter returns a safe `tool_calling_unsupported` error rather than
retrying or switching providers. Both adapters enforce the configured model and
`LLM_REQUEST_TIMEOUT` (1–120 seconds). OpenAI SDK retries are disabled; the adapters convert
authentication, rate-limit, timeout, connection, HTTP, and malformed-response failures into
sanitized provider errors without returning raw provider details.

Configure these variables in the process environment before creating a provider. Set exactly one
provider; do not put credentials in source control or command history. For example, an OpenAI key
can be entered using PowerShell's hidden prompt:

```powershell
$env:LLM_PROVIDER = "openai"
$env:OPENAI_MODEL = "your-approved-model"
$secureApiKey = Read-Host "OpenAI API key" -AsSecureString
$credential = [System.Management.Automation.PSCredential]::new("OpenAI", $secureApiKey)
$env:OPENAI_API_KEY = $credential.GetNetworkCredential().Password
Remove-Variable secureApiKey, credential
$env:LLM_REQUEST_TIMEOUT = "30"
```

For Ollama, use `$env:LLM_PROVIDER = "ollama"`, set `$env:OLLAMA_MODEL` to an already available
model, and optionally set `$env:OLLAMA_BASE_URL`. The Ollama model must support tool calling to
answer tool-based requests. Leaving `LLM_PROVIDER` unset leaves the assistant unavailable; an
explicitly selected provider with missing configuration fails when provider creation is attempted.

Provider adapters only translate messages, tool definitions, responses, and tool results; they do
not execute tools. Provider output must still be validated by the fixed local tool registry and
trusted executor. Calls transmit the supplied messages, tool definitions, and tool results to the
selected endpoint: OpenAI's API or the exact configured Ollama URL, which may be remote. Adapters
do not read or append database readings, credentials, or infrastructure details, but they cannot
inspect arbitrary message content. The M6.2 orchestrator and its callers must not put raw household
readings or secrets into provider requests. Adapter tests use fake clients and do not call OpenAI,
require an Ollama server, or download models.

Run the offline adapter tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_assistant_providers.py
```

### M6.1 Step 2: read-only tool schemas and registry

The immutable registry exposes exactly these operations:

| Tool | Inputs | Structured output |
|---|---|---|
| `get_energy_summary` | Meter, start, end | Reading count, active-power average/min/max/latest, and missing-measurement counts/rates |
| `get_energy_timeseries` | Meter, start, end, `hour` or `day` | Bounded buckets with reading counts and supported measurement averages/min/max |
| `get_electrical_metrics` | Meter, start, end | Reading count and per-measurement non-null count, average, min/max, missing count/rate |
| `get_data_quality` | Meter, start, end | Missing measurements, observed gaps, estimated missing intervals, and explicit data-availability limitations |
| `get_anomaly_summary` | Meter, run ID, or both | Run status, processed/normal/anomaly/unavailable counts, anomaly rate, category/algorithm counts |
| `get_detection_run_details` | Run ID | Run identity, meter, algorithm, status, timestamps, and prediction counts |
| `compare_energy_periods` | Meter and start/end for both periods | Per-period reading count and active-power average/min/max; B-minus-A and percent changes |

Tool arguments are validated locally by strict Pydantic v2 schemas: unexpected fields are
rejected, meter identifiers follow the API's 1–100 character constraint, timestamps must include
a UTC offset and are normalized to UTC, and analytics periods are bounded to 31 days. All date
intervals use `[start, end)`. Time-series intervals are only `hour` or `day`, with a maximum of
750 buckets. Period comparisons reject overlap. The registry performs exact-name lookup and
argument validation only; it has no execution or dynamic handler-dispatch capability.

The registry is deliberately read-only. Arbitrary SQL/Python, shell or filesystem access,
database writes, meter/reading mutations, detection-job submission, configuration access, and
environment inspection are not registered.

Existing API behavior and data limitations remain authoritative. Electrical readings contain
the seven persisted columns (`global_active_power`, `global_reactive_power`, `voltage`, `global_intensity`, and
`sub_metering_1` through `sub_metering_3`). Electrical range-wide
statistics use direct SQL aggregates over observations, not averages of time buckets. Quality
data is computed from stored readings only: gap estimates assume one-minute sampling, boundary
gaps are not assessed, and ingestion-time findings cannot be reconstructed. Anomaly summaries
preserve the distinctions between unknown meters/runs, no completed run, non-completed runs, and
completed runs with empty anomaly results. Detection-run tool results omit internal configuration
and worker details. Period comparisons aggregate each interval independently using the existing
summary query.

### M6.1 Step 3: trusted read-only execution

`AnalyticsToolExecutor` now runs the seven registered tools synchronously. Each request is
validated against the fixed Pydantic schema before a short-lived SQLAlchemy session is opened;
one session context is closed on success or exception. Dispatch uses fixed in-code name checks,
never a supplied function, module, SQL statement, or handler. Results are projected to JSON-safe
objects. Invalid arguments, unknown tools/meters/runs, missing completed runs, non-completed run
summaries, database failures, unexpected internal errors, and unsafe serialization failures have
distinct safe error codes.

The tools return bounded summary/time-series, electrical measurement aggregates, observed data
quality, completed-run anomaly counts, selected run details, and two-period energy comparisons.
Electrical averages/minima/maxima are SQL aggregates over the underlying observations. Period
comparisons run the same summary aggregation independently for each half-open interval and return
period B minus period A differences. Percentage change is `(B - A) / A * 100`; it is `null` if
the baseline is zero or either measurement is unavailable. Reading-count differences are also
reported.

Detection-run details intentionally omit configuration, worker lease/heartbeat fields, retry
attempts, and internal error text. Anomaly summaries preserve the distinction between an unknown
meter, no completed run, unknown run, and a known non-completed run. Quality output remains based
only on database-observed data; timestamp-gap estimates assume one-minute sampling, do not assess
boundary gaps, and cannot reconstruct ingestion-time validation.

The execution layer does not submit detection jobs or expose mutation, SQL, Python, shell,
filesystem, configuration, or environment-inspection capabilities. Orchestration, assistant API,
conversation persistence, and chat UI remain unimplemented; this is not yet a user-facing LLM
assistant.

## Current scope

Milestones 1 and 2 cover ingestion, quality reporting, Parquet output, PostgreSQL persistence,
migrations, and batch import. Milestone 3 includes statistical detectors, past-only feature
engineering, controlled synthetic evaluation data, chronological evaluation, Isolation Forest,
and the end-to-end detection pipeline. Milestone 4 currently includes the FastAPI health/readings
API, durable detection jobs, the standalone PostgreSQL worker, and persisted anomaly queries.
Milestone 5 provides the analytics API and all six Streamlit dashboard sections, backed by shared
meter/date filters and an HTTPX client. M6.1 Steps 1–3 establish optional provider configuration,
provider-neutral contracts, a read-only analytics tool registry, and bounded trusted execution;
provider adapters, orchestration, assistant API, and chat UI remain future work. PostgreSQL
pipeline input requires a configured database. The API is local-only and
has no authentication; authentication/authorization, global rate limiting, and complete Docker/CI
deployment remain future work.
