# Pytorch Profiling

By example profiling of pytorch training.

Goal: train a ViT on Flowers102 dataset, diagnosing situations where we are IO bound, CPU-bound, other

install:

```bash
python -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -e .
```

dev:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Run

Review `configs/training.yaml` before running

```bash
   python runner.py fit -c configs/training.yaml --trainer.logger.name my_run_name
```
