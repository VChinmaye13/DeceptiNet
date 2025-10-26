#!/usr/bin/env bash
set -e

python -m src.detector.train_detector --config config.yaml "$@"
python -m src.detector.eval_detector --config config.yaml "$@"
python app/gradio_app.py --config config.yaml "$@"
