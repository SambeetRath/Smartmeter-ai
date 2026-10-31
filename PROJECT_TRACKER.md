# SmartMeter AI — Project Tracker

**Project:** SmartMeter AI — Energy Data Analytics & Anomaly Detection Platform  
**Repository:** [SambeetRath/Smartmeter-ai](https://github.com/SambeetRath/Smartmeter-ai)  
**Project status snapshot:** 31 October 2026  
**Current milestone:** Milestone 4 — REST API Development  
**Current phase:** Milestone 4 — REST API Development  
**Overall status:** In Progress — Milestone 3 complete

---

## 1. Project Overview

SmartMeter AI processes smart-meter energy readings, identifies unusual consumption patterns,
stores and exposes energy data, and is planned to provide analytical insights through a dashboard
and an LLM-powered assistant.

### Technology stack

| Area | Technology |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Database | PostgreSQL |
| ORM and migrations | SQLAlchemy, Alembic |
| Machine learning | scikit-learn |
| Backend | FastAPI (planned) |
| Dashboard | Streamlit, Plotly (planned) |
| LLM integration | LLM tool calling (planned) |
| Testing | pytest; HTTPX planned |
| Code quality | Ruff |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |

### Dataset

The UCI Individual Household Electric Power Consumption dataset represents one household and
contains minute-level readings: 2,075,259 observations covering 16 December 2006 through
26 November 2010. Missing measurement values are preserved and handled explicitly.

Synthetic anomaly labels are for controlled evaluation only; they are not verified real-world
anomaly ground truth.

---

## 2. Overall Milestone Tracker

| Milestone | Description | Status | Completion date |
|---|---|---|---|
| M1 | Data ingestion and validation | Complete | Not recorded |
| M2 | PostgreSQL persistence | Complete | Not recorded |
| M3 | Anomaly detection and machine learning | Complete | 31 Oct 2026 |
| M4 | REST API development | Planned | — |
| M5 | Analytics dashboard | Planned | — |
| M6 | LLM-powered assistant | Planned | — |
| M7 | Docker, CI/CD and deployment | Planned | — |

---

## 3. Milestone 1 — Data Ingestion and Validation

**Status: COMPLETE**

### Implemented

- UCI dataset ingestion, cleaning, and normalization.
- Timestamp parsing and chronological sorting.
- Explicit missing-value handling.
- Data-quality validation, including duplicate timestamp and exact-record checks.
- Missing-interval detection.
- Negative and non-finite value detection.
- Exploratory IQR outlier detection.
- JSON quality report and processed Parquet output.

### Recorded verification

- The original dataset was preserved.
- 2,075,259 rows were processed.
- Missing values were retained in processed output.
- Parquet output was verified against its source.
- Unit tests passed.

---

## 4. Milestone 2 — PostgreSQL Persistence

**Status: COMPLETE — LIVE DATABASE VERIFICATION PREVIOUSLY PASSED**

### Implemented

- PostgreSQL 17 using Docker Compose.
- SQLAlchemy database models for meters and meter readings.
- Alembic migration and repository layer.
- Batched Parquet ingestion with idempotent insertion.
- NULL preservation and UTC timestamp handling.

### Previous live database verification

- 2,075,259 records inserted.
- Repeat import skipped all existing records.
- No duplicate meter/timestamp pairs were found.
- Missing values in all seven measurement columns were preserved.
- Alembic migration applied successfully.
- Integration tests previously passed against a live database.

### Outstanding verification

Run PostgreSQL integration tests again before final project release.

---

## 5. Milestone 3 — Anomaly Detection and Machine Learning

**Status: IN PROGRESS**

### Phase 1 — Architecture Inspection

**Status: COMPLETE**

- Existing architecture, ingestion, and validation APIs were inspected.
- Anomaly detection implementation was planned.

### Phase 2 — Statistical Anomaly Detection

**Status: COMPLETE**

Implemented global IQR, rolling mean/standard-deviation, and time-of-day-aware statistical
detectors. The time-dependent baselines use past observations, handle missing values, and validate
timestamps. These are exploratory detectors, not verified anomaly labels.

### Phase 3 — Feature Engineering and Synthetic Data

**Status: COMPLETE**

Implemented temporal and cyclical features, configurable lag and historical rolling features,
past-only feature calculations, reproducible synthetic consumption spikes, low-consumption,
sustained-consumption and voltage anomalies, and chronological train/validation/test splitting.

Synthetic labels are explicitly separate from real-world ground truth.

### Phase 4 — Isolation Forest

**Status: COMPLETE — VERIFIED 31 OCTOBER 2026**

Implemented in `src/smartmeter/anomaly_detection/isolation_forest.py`:

- Configurable scikit-learn Isolation Forest estimator.
- Configurable and validated feature selection and model parameters.
- Median imputation fitted only on the supplied training frame.
- Training fit with label-like columns rejected.
- Prediction labels and anomaly scores aligned to copied input rows and indices.
- Atomic model save/load utilities.
- Unit tests for configuration, training-only imputation, row preservation, missing/non-finite
  features, label rejection, and artifact round-trip.

`scikit-learn==1.9.1` was added to `pyproject.toml` and `requirements.txt`. The detector produces
unsupervised model decisions, not verified anomaly labels or calibrated probabilities.

### Phase 5 — Evaluation and Comparison

**Status: COMPLETE — VERIFIED 31 OCTOBER 2026**

Implemented:

- `src/smartmeter/anomaly_detection/evaluation.py` provides validated binary metrics, coverage
  and exclusion accounting, chronological evaluation, category analysis, contamination sweeps,
  and strict JSON report writing.
- `GlobalIQRDetector` in `statistical.py` shares the existing 1.5×IQR threshold rule, fits
  bounds on training observations, and applies those bounds to held-out partitions. The
  existing `detect_global_iqr` exploratory behavior remains available.
- The evaluation compares global IQR, rolling mean/std, time-of-day, and Isolation Forest on
  validation and test synthetic copies. A shared-row comparison reports all detectors on their
  common available test observations.
- Reports include TP/TN/FP/FN, precision, recall, F1, accuracy, false-positive rate, specificity,
  confusion matrices in `[normal=0, anomaly=1]` order, coverage, excluded rows, synthetic label
  counts, and per-category metrics.
- Category analysis is one-vs-rest. Other anomaly categories are reported separately from normal
  observations; unknown categories are counted rather than silently discarded.
- Undefined zero-denominator metrics use the explicit `0.0` policy. A detector with no available
  observations reports `null` metrics. JSON serialization rejects NaN and infinity.
- Optional contamination candidates are independently fitted on training data and compared on
  validation only. The test model uses the explicitly configured contamination; no candidate is
  automatically selected from test performance.
- Synthetic injection occurs only in validation/test copies. Isolation Forest imputation and
  fitting use training rows only; synthetic labels are not model features or training labels.
- `README.md` documents methodology, limitations, report generation and execution.

Files created:

- `src/smartmeter/anomaly_detection/evaluation.py`
- `tests/unit/test_evaluation.py`

Files modified:

- `src/smartmeter/anomaly_detection/statistical.py`
- `src/smartmeter/anomaly_detection/__init__.py`
- `README.md`
- `PROJECT_TRACKER.md`

No dependency declarations changed for Phase 5.

### Phase 6 — End-to-End Pipeline and Reporting

**Status: COMPLETE — VERIFIED 31 OCTOBER 2026**

Implemented in `src/smartmeter/anomaly_detection/pipeline.py` with the package CLI in
`src/smartmeter/anomaly_detection/__main__.py`:

- Reads normalized Parquet datasets or an existing PostgreSQL meter through ordered bounded-page
  reads. The database reader materializes the full series in memory for current batch processing.
- Validates the normalized timestamp/seven-measurement schema, rejects invalid or duplicate
  timestamps, invalid numeric values, and label-like input columns. Missing values and all valid
  source rows are retained; input data is not modified.
- Configures and runs global IQR, rolling mean/std, time-of-day, and Isolation Forest detectors
  independently or together. Statistical detectors support selected measurements and thresholds.
- Uses one chronological training/held-out split. Global IQR thresholds and Isolation Forest
  preprocessing/model fitting use training rows only. Rolling/time-of-day baselines use only
  earlier observations, including legitimate training history. Synthetic labels are rejected.
- Emits a consistent sparse CSV result schema containing alerts and unavailable predictions;
  available normal predictions are counted in JSON but omitted from CSV. Detector metadata,
  score semantics, run ID, source, time boundaries, coverage, exclusions, and configurations are
  recorded in the JSON summary. Detector alerts are not verified real-world labels.
- Protects existing output files unless `--overwrite` is explicitly supplied. CLI supports
  Parquet/PostgreSQL input, detector selection, measurement selection, split fraction, and
  statistical thresholds.
- Added `joblib==1.6.0` to both dependency manifests because the existing Isolation Forest module
  imports it directly for model persistence.

Created:

- `src/smartmeter/anomaly_detection/pipeline.py`
- `src/smartmeter/anomaly_detection/__main__.py`
- `tests/unit/test_pipeline.py`

Modified:

- `src/smartmeter/anomaly_detection/__init__.py`
- `pyproject.toml`
- `requirements.txt`
- `README.md`
- `PROJECT_TRACKER.md`

### Phase 7 — Documentation and Final Verification

**Status: COMPLETE — VERIFIED 31 OCTOBER 2026**

- Updated README with the pipeline workflow, supported inputs, output schema, leakage safeguards,
  score interpretation, CLI examples, overwrite protection, and limitations.
- Updated this tracker from actual test and runtime output. Future API, dashboard, LLM, and
  deployment milestones remain planned.
- Confirmed `pyproject.toml` runtime dependencies match `requirements.txt`; editable installation,
  dependency checks, Ruff, focused formatting, and `git diff --check` passed.
- Verified the raw dataset, processed Parquet, and generated reports remain excluded by
  `.gitignore`.
- Full-data Parquet runtime and PostgreSQL integration tests passed. A PostgreSQL-source pipeline
  smoke test used an isolated two-row synthetic test meter; full-scale PostgreSQL pipeline memory
  and runtime remain unmeasured.
- Dockerfile/Compose boundaries were inspected. No Docker image build was performed because
  container deployment is outside Milestone 3.

---

## 6. Remaining Milestones

### Milestone 4 — REST API Development

**Status: PLANNED**

Expected scope: FastAPI application, health check, meter and energy-consumption endpoints,
time-range filtering, anomaly prediction endpoints, request/response validation, error handling,
API tests, and OpenAPI documentation.

### Milestone 5 — Analytics Dashboard

**Status: PLANNED**

Expected scope: Streamlit dashboard, energy trends, daily/hourly analysis, missing-data and anomaly
visualization, detector comparisons, interactive filters, and Plotly charts. The UCI dataset
contains one household; multi-meter comparisons require separately labelled synthetic or
additional real datasets.

### Milestone 6 — LLM-Powered Assistant

**Status: PLANNED**

Expected scope: natural-language questions, tool-based backend integration, grounded consumption
summaries, anomaly explanations, and time-range queries without invented readings.

### Milestone 7 — Docker, CI/CD and Deployment

**Status: PLANNED**

Expected scope: application containerization, Docker Compose services, environment-based
configuration, GitHub Actions, automated tests/linting, deployment preparation, security review,
and final demonstration.

---

## 7. Verification History

| Date | Scope | Result |
|---|---|---|
| 30 Oct 2026 | Milestone 2 live database verification | Passed |
| 31 Oct 2026 | Milestone 3 Phase 3 verification | 74 passed, 2 skipped |
| 31 Oct 2026 | Milestone 3 Phase 4 full project test suite | 88 passed, 2 skipped |
| 31 Oct 2026 | Ruff (`src tests`) | Passed |
| 31 Oct 2026 | `git diff --check` | Passed |
| 31 Oct 2026 | `pip check` | No broken requirements found |
| 31 Oct 2026 | Editable install with `[dev]` | Succeeded; scikit-learn 1.9.1 installed |
| 31 Oct 2026 | Milestone 3 Phase 5 focused tests | 25 passed in 6.25s |
| 31 Oct 2026 | Milestone 3 Phase 5 full project checks | 113 passed, 2 skipped in 7.55s; Ruff and diff check passed |
| 31 Oct 2026 | Phase 5 dependency check | No broken requirements found |
| 31 Oct 2026 | Phase 5 synthetic sample benchmark | 4,000 rows; train/validation/test 2,800/600/600; 1.763s; source frame 256,132 bytes |
| 31 Oct 2026 | Editable install after dependency update | `pip install -e ".[dev]"` succeeded; joblib 1.6.0 already satisfied |
| 31 Oct 2026 | Phase 6 focused pipeline tests | 21 passed in 4.22s |
| 31 Oct 2026 | `check_project.ps1` | 134 passed, 2 skipped in 9.31s; Ruff and `git diff --check` passed |
| 31 Oct 2026 | PostgreSQL integration tests on `smartmeter_test` | 2 passed in 3.44s |
| 31 Oct 2026 | Complete pytest suite with `SMARTMETER_TEST_DATABASE_URL` | 136 passed in 9.16s |
| 31 Oct 2026 | Ruff check and focused Ruff format check | Passed |
| 31 Oct 2026 | `pip check` and dependency manifest comparison | No broken requirements; runtime declarations matched |
| 31 Oct 2026 | PostgreSQL-source pipeline smoke test | 2 synthetic meter rows read; one held-out row and both outputs created |
| 31 Oct 2026 | Full processed-Parquet pipeline run | 2,075,259 input; 1,452,681 train; 622,578 held out; 46.574s; 219,236 sparse results; CSV 84,568,657 bytes |
| 31 Oct 2026 | Final result artifact validation | CSV schema, counts, nullable unavailable predictions, and chronological ordering passed |

`check_project.ps1` was run without `SMARTMETER_TEST_DATABASE_URL`, so its two PostgreSQL tests
were skipped in that invocation. They were then run against the isolated `smartmeter_test`
database and passed; the complete suite with that variable configured passed all 136 tests.

The 4,000-row benchmark used controlled synthetic data, not the UCI dataset. It generated
28 labeled test observations (6 spikes, 6 unusually low, 10 sustained, 6 voltage). Test coverage
was 100% for all four detectors. Global IQR, rolling mean/std, and time-of-day each had precision
1.0000, recall 0.4643, and F1 0.6341 on this sample; Isolation Forest had precision 0.1481,
recall 0.1429, and F1 0.1455. These are sample-only results and are not estimates of real-world
performance.

---

## 8. Current Action Items

- Begin Milestone 4 — FastAPI development.
- Preserve the pipeline's chronological split, training-only fitted preprocessing, and no-label
  inference boundary when adding API endpoints.
- Before release, consider a full-scale PostgreSQL-source pipeline benchmark and peak-memory
  measurement.

### Milestone 3 limitations and remaining checks

- Evaluation was measured only on a 4,000-row controlled synthetic sample; the full UCI dataset
  was not used for labeled accuracy evaluation. The Phase 6 full-data run was inference only and
  provides no precision/recall evidence.
- Peak process memory was not measured. PostgreSQL-source smoke testing used a two-row synthetic
  meter; full-scale PostgreSQL processing is not benchmarked and currently materializes readings
  in memory.
- Synthetic labels do not establish real-world detection accuracy. No detector is claimed to be
  superior based on the sample results or the unlabeled UCI inference run.
- Global IQR uses a train-fitted global distribution and does not model time-of-day or seasonal
  changes; rolling and time-of-day coverage depends on historical warm-up.
- Isolation Forest does not model temporal ordering except through supplied past-only features.
- `pydantic` remains in the existing runtime dependency manifests but is not imported by current
  source modules; it was retained to avoid unrelated dependency cleanup.
- The repository had existing uncommitted changes before this task. They were preserved; no
  commit or push was made.

---

## 9. Development Log

| Date | Milestone / Phase | Work completed | Verification | Notes |
|---|---|---|---|---|
| 31 Oct 2026 | M3 — Phase 4 | Isolation Forest implementation | 88 passed, 2 skipped; Ruff and diff check passed | Complete |
| 31 Oct 2026 | M3 — Phase 5 | Leakage-aware detector evaluation, metrics, category analysis, JSON report writer | 25 focused passed; full suite 113 passed, 2 skipped; Ruff, pip check, and diff check passed | Complete; 4,000-row synthetic benchmark only |
| 31 Oct 2026 | M3 — Phases 6–7 | Integrated configurable anomaly pipeline, CSV/JSON reporting, documentation, and final verification | 136 passed; PostgreSQL integration 2 passed; Ruff, focused format, pip check, and diff check passed; full Parquet run 46.574s | Complete; full PostgreSQL scale and peak memory not benchmarked |

Add a new row after every approved phase.

---

## 10. Rules for Updating This Tracker

1. Update status only after implementation and verification.
2. Never mark a phase complete solely because code was generated.
3. Record actual test counts and explain skipped tests.
4. Record dependency changes and unresolved issues.
5. Update the current phase and next action.
6. Do not invent completion dates.
7. Preserve historical entries instead of overwriting verification results.
8. Update this document after each approved phase and milestone.

**Last updated:** 31 October 2026  
**Current next task:** Milestone 4 — REST API Development
