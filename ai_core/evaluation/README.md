# SafeHER AI/ML — Evaluation Framework

## Purpose
This directory hosts benchmark datasets, evaluation protocols, and validation scripts for SafeHER threat assessment and digital abuse detection pipelines.

## Evaluation Principles
1. **No Fabricated Benchmarks**: Evaluation metrics (Precision, Recall, F1, AUROC) must only be reported when computed against real, verified test runs on benchmark datasets.
2. **Synthetic / Sanitized Evaluation Data**: Never use unredacted real victim records or sensitive personal data for testing or model benchmarking.
3. **Calibration & Uncertainty**: Evaluate models not only on accuracy but also on Expected Calibration Error (ECE) and uncertainty reliability under out-of-distribution inputs.
4. **Figurative & Idiomatic Robustness**: Benchmarks include stress tests with common idioms, slang, and figurative speech to measure and minimize false positive rates.
