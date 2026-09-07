mlops-foundations-to-production
==============================

# MLOps From Scratch

A structured, end-to-end walkthrough of the MLOps stack : Built while learning, one component at a time, from raw data to a monitored production-style deployment.

This repo is not a single polished pipeline; it's a comprehensive reference covering the full MLOps lifecycle, with each major topic implemented and documented as its own module.

## What's about to be covered (In next few days)

- **Data Versioning** : DVC for tracking datasets and reproducible data pipelines
- **Experiment Tracking** : logging runs, metrics, and hyperparameters (MLflow / W&B)
- **Model Registry** : versioning and staging trained models (staging -> production)
- **Model Serving** : wrapping trained models in an API for inference
- **CI/CD** : automated testing and deployment pipelines with GitHub Actions
- **Docker** : containerizing the training and serving environments
- **Monitoring** : Prometheus for metrics collection, Grafana for dashboards, tracking model/service health in production


2. dvc stage add -n data_preprocessing -d "1. data-versioning/src/pre_process.py" -d data/raw -o data/processed python "1. data-versioning/src/pre_process.py"

3. dvc stage add --force -n feature_engineering -d "1. data-versioning/src/feature_engineering.py" -d data/processed -o data/features python "1. data-versioning/src/feature_engineering.py"