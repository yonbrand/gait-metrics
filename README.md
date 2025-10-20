# Gait Metrics

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

`gait-metrics` is a Python package and command-line tool for computing digital gait metrics from raw accelerometer data. It uses a series of pre-trained deep learning models (`ElderNet`) to analyze acceleration signals and estimate metrics like gait speed, cadence, step count, and stride regularity.

The models are downloaded automatically on first use and cached locally in `~/gait_metrics/models/`.

## Features

* **Multi-File Processing:** Handles various accelerometer formats, including `.cwa`, `.gt3x`, `.bin`, and `.csv`.
* **Comprehensive Metrics:** Computes a suite of gait outcomes:
    * Walking Time
    * Step Count
    * Gait Speed
    * Cadence
    * Stride Length (model-based and indirect)
    * Gait Regularity (model-based and signal processing-based)
* **Data Preprocessing:** Includes built-in support for resampling, non-wear detection, and filtering data based on wear time.
* **Bout Analysis:** Optionally segments walking into bouts and calculates bout-specific statistics.

## Installation

To install the package, clone this repository and install it locally using `pip`.

```bash
# 1. Clone the repository
git clone [https://github.com/yonbrand/gait-metrics.git](https://github.com/yonbrand/gait-metrics.git)
cd gait-metrics

# 2. Install the package
# This will also install all dependencies listed in pyproject.toml
pip install .
