# Point Transformer V3 (PT-v3) - FVDB Implementation

This repository contains a minimal implementation of Point Transformer V3 using the FVDB library for scalable 3D point cloud processing.

## Setup

```bash
# Activate fvdb conda environment
conda activate fvdb

# Install dependencies
cd fvdb-examples/point_transformer_v3
pip install -r requirements.txt
```



## Quick Test

```bash
# Download test data
python scripts/data/download_example_data.py

# Run inference
python scripts/test/minimal_inference.py --data-path data/scannet_samples_small.json --voxel-size 0.1 --patch-size 1024 --batch-size 1

# Compare results
python scripts/test/compute_difference.py --stats_path_1 data/scannet_samples_small_output.json --stats_path_2 data/scannet_samples_small_output_gt.json
```


## Project Structure

- `fvdb_extensions/models/ptv3_fvdb.py` - Core FVDB implementation
- `fvdb_extensions/models/point_transformer_v3m1_fvdb.py` - Pointcept framework adapter
- `scripts/data/` - Data download and preprocessing scripts
- `scripts/test/` - Inference and comparison scripts

