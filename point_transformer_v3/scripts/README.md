# Scripts Directory

This directory contains utility scripts organized by purpose.

## `data/` - Data Management Scripts

Scripts for downloading and preprocessing datasets:

- **`download_example_data.py`**: Downloads preprocessed test data from remote repository
- **`prepare_scannet_dataset.py`**: Prepares ScanNet dataset samples from raw data

## `test/` - Testing and Validation Scripts

Scripts for running inference and validating results:

- **`minimal_inference.py`**: Runs PT-v3 model inference on point cloud data
- **`compute_difference.py`**: Compares inference outputs between different implementations

## Usage

All scripts should be run from the `point_transformer_v3/` directory:

```bash
# Data scripts
python scripts/data/download_example_data.py
python scripts/data/prepare_scannet_dataset.py --data-root /path/to/scannet --output data/samples.json

# Test scripts
python scripts/test/minimal_inference.py --data-path data/scannet_samples.json
python scripts/test/compute_difference.py --stats_path_1 data/output1.json --stats_path_2 data/output2.json
```

