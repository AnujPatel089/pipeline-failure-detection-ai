# Pipeline Failure Detection AI

[![CI](https://github.com/AnujPatel089/pipeline-failure-detection-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/AnujPatel089/pipeline-failure-detection-ai/actions/workflows/ci.yml)

Pipeline Failure Detection AI is a two-stage machine-learning system for SCADA telemetry monitoring and replay. It first detects abnormal pipeline conditions, then classifies detected faults as blockage, degradation, leak, or surge. A shared inference layer powers both the FastAPI service and the three-workspace Streamlit dashboard, while leakage-safe features and group-aware validation keep evaluation boundaries credible.

## Project Highlights

- Two-stage failure detection and conditional fault classification.
- Leakage-safe design restricted to eight approved telemetry features.
- Group-aware evaluation that prevents pipeline segments from crossing split boundaries.
- Interactive Streamlit UI with three operational workspaces and four themes.
- FastAPI endpoints for single and batch inference.
- Standard (`0.50`) and High Sensitivity (`0.30`) operating thresholds.
- 85 automated tests with deterministic SCADA-like fixtures.
- GitHub Actions CI for every push and pull request.
- Raw Kaggle data excluded from the repository and ordinary test suite.

## What It Does

```text
SCADA telemetry → input validation → binary failure detection → threshold policy
                                                               ├─ normal → healthy
                                                               └─ abnormal → fault classification
                                                                                 ↓
                                                                     FastAPI / Streamlit
```

The application replays stored observations by pipeline segment, validates telemetry, predicts normal or abnormal operation, and invokes fault classification only for an abnormal result. It supports two monitoring modes, exposes structured API predictions, and presents operational context in the dashboard. This is real-time-style replay, not a live industrial SCADA connection.

## Application Preview

Real application screenshots have not yet been added. The following files are reserved for future captures in `docs/images/`:

- `dashboard-overview.png` — Operations Overview
- `dashboard-alert.png` — Critical Alert State
- `investigation-view.png` — Telemetry Investigation
- `api-swagger.png` — FastAPI Swagger UI

<!-- Activate these sections only after the corresponding real screenshots exist.

### Operations Overview

![Operations Overview](docs/images/dashboard-overview.png)

### Critical Alert State

![Critical Alert](docs/images/dashboard-alert.png)

### Telemetry Investigation

![Telemetry Investigation](docs/images/investigation-view.png)

### FastAPI

![FastAPI Swagger UI](docs/images/api-swagger.png)
-->

## System Architecture

```mermaid
flowchart TD
    A[SCADA Telemetry] --> B[Input Validation]
    B --> C[Shared Inference Service]
    C --> D[Binary Failure Detector]
    D --> E[Threshold Policy]
    E -->|Normal| F[Healthy]
    E -->|Abnormal| G[Fault Classifier]
    G --> H[Blockage / Degradation / Leak / Surge]
    F --> I[Structured Prediction]
    H --> I
    I --> J[FastAPI]
    I --> K[Streamlit]
```

Both fitted artifacts contain preprocessing and classification in one scikit-learn pipeline. Models are loaded once per application lifecycle, and both presentation layers use shared inference rather than duplicating transformations or prediction logic.

## Model Performance

### Binary failure detector

Official held-out pipeline segments:

| Metric | Result |
|---|---:|
| Failure precision | 88.5% |
| Failure recall | 93.1% |
| F1 | 90.8% |
| PR-AUC | 97.2% |

The held-out evaluation detected 54 of 58 failures, with 4 false negatives and 7 false positives.

Repeated segment-group validation:

| Metric | Mean ± standard deviation |
|---|---:|
| Failure recall | 88.1% ± 4.1% |
| F1 | 89.6% ± 3.2% |

### Fault classifier

| Evaluation | Result |
|---|---:|
| Official held-out macro F1 | 97.3% |
| Official balanced accuracy | 96.9% |
| Group-aware CV macro F1 | 97.2% ± 2.0% |

The weakest held-out class was blockage, with 87.5% recall. The only held-out fault-classification error was a blockage observation classified as a leak. Detailed validation summaries, error analyses, feature importance, and plots are retained under `artifacts/`. These results come from a small simulated-style dataset and are not production-performance claims.

## Leakage Prevention

The approved predictive inputs are:

```text
pressure, flow_rate, temperature, valve_status,
pump_state, pump_speed, compressor_state, energy_consumption
```

The following fields are forbidden as model features:

- `target`: prediction label.
- `event_type`: directly encodes normal and fault categories.
- `alarm_triggered`: strongly encodes an existing alarm outcome.
- `timestamp`: ordering metadata only.
- `segment_id`: grouping and response metadata only.

Pydantic rejects labels and alarm fields at the API boundary. Dashboard evaluation mode attaches ground truth only after prediction. Automated tests verify that inference receives exactly the eight approved fields, and group-aware splits prevent segments from crossing train/test boundaries.

## Monitoring Modes

| Mode | Threshold | Precision | Recall | Tradeoff |
|---|---:|---:|---:|---|
| Standard | 0.50 | 92.5% | 89.5% | Fewer false alerts |
| High Sensitivity | 0.30 | 76.4% | 95.2% | More failures detected, more false alerts |

Threshold selection used out-of-fold training predictions rather than final test labels. The alternate mode changes only the decision threshold; it does not modify the saved model.

## Dashboard

The Streamlit dashboard has three workspaces:

1. Operations Overview
2. Telemetry & Investigation
3. Model & System

They share navigation and session state for segment replay, timestamp position, monitoring mode, prediction history, and theme selection. Available themes are Industrial Slate, Deep Navy, Light Operations, and Steel Blue. An optional evaluation panel shows ground truth only after inference.

## FastAPI

Interactive documentation is available at `http://127.0.0.1:8000/docs` and ReDoc at `http://127.0.0.1:8000/redoc`.

- `GET /health` — process liveness
- `GET /ready` — model-service readiness
- `GET /model-info` — validated safe metadata
- `POST /predict` — single observation
- `POST /predict/batch` — up to 1,000 observations

Every response carries an `X-Request-ID`. Application logs are structured JSON and avoid telemetry payloads, model contents, and local filesystem paths.

## Quick Start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch the dashboard:

```bash
python -m streamlit run app.py
```

Start the API:

```bash
python -m uvicorn api.main:app --reload
```

Run the automated checks:

```bash
python -m compileall src api
python -m pytest -q
```

The ordinary test suite is self-contained and does not require the external Kaggle dataset.

## Tests

The 85-test suite covers validation, leakage boundaries, saved-model integration, inference behavior, FastAPI, operational quality, and Streamlit dashboard behavior. Dashboard tests use the deterministic `scada_frame` fixture: 250 schema-valid SCADA-like observations across 50 segments. Tests do not read `data/raw/` or redistribute Kaggle data.

## CI

GitHub Actions validates the project on every push and pull request using Python 3.11. The workflow:

1. Checks out the repository.
2. Installs dependencies from `requirements.txt`.
3. Compiles `src` and `api` with `python -m compileall src api`.
4. Runs all 85 tests with `python -m pytest -q`.

The CI suite is self-contained and does not require the external Kaggle dataset, Kaggle credentials, or a dataset download. Dashboard tests use deterministic SCADA-like fixtures. CI validates the codebase; it does not perform deployment.

## Dataset

This project uses the [SCADA Pipeline Operations Dataset](https://www.kaggle.com/datasets/zara2099/scada-pipeline-operations-dataset) from Kaggle. Raw data is not redistributed in this repository. The external dataset is needed only for full dataset verification and real-data replay where applicable; users should obtain it from Kaggle and review the applicable terms there. No dataset license is asserted by this project.

For full dataset verification, place the CSV under `data/raw/`, then run:

```bash
python -m src.full_dataset_verification
```

This checks schema, 1,000 rows, 50 segments, 17 timestamps, target/event distributions, and compatibility with both saved pipelines. It does not retrain either model.

## Project Structure

```text
api/                       FastAPI application, schemas, middleware, errors
artifacts/                 Final evaluation tables, reports, and plots
data/raw/                  External Kaggle CSV; ignored by Git
docs/images/               Reserved for future real screenshots
models/                    Two fitted pipelines and safe metadata
src/                       Training, validation, inference, and services
tests/                     Deterministic software and model integration tests
.github/workflows/ci.yml   Python 3.11 compile and pytest workflow
app.py                     Streamlit dashboard
requirements.txt           Direct project dependencies
```

## Limitations

- The dataset contains 1,000 observations, including 306 abnormal records.
- It covers 50 segments and 17 timestamps—approximately 17 minutes of observations.
- The data has simulated/synthetic-style characteristics and unusually separable telemetry.
- Strong results may partly reflect data-generation artifacts.
- There is no live industrial SCADA integration or prospective field validation.
- There is no authentication, production rate limiting, or distributed serving layer.
- This is a portfolio prototype, not an operational safety system.

## Tech Stack

Python 3.11, Pandas, NumPy, scikit-learn, Joblib, FastAPI, Pydantic, Uvicorn, Streamlit, Matplotlib, Pytest, and GitHub Actions.
