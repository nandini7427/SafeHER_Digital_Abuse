# Evaluation scripts and generated outputs

This directory contains evaluation utilities and generated reports. Generated outputs must always be interpreted with their generation script, dataset source, run configuration and limitations.

In particular, `per_image_evaluation_results.csv` is a controlled/fixture benchmark artifact from the repository's evaluation harness. It is **not** an independent real-world test set and does not measure the performance of full-scale trained Model A or Model B checkpoints. The M4 checkpoints needed for such a benchmark were not supplied in this repository. Do not quote this CSV as production or survivor-case accuracy.

`benchmark_results.json` contains host-specific measurements and must be regenerated on the deployment target before making resource/latency claims. `sbom.json` is a generated software inventory snapshot; regenerate it when dependencies or the build environment change.
