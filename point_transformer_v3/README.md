# Point Transformer V3 (PT-v3) - FVDB Implementation

This example trains and runs Point Transformer V3 with `fvdb.nn.PointTransformerV3` from fVDB, through Pointcept.

## Setup

```bash
# Activate fvdb conda environment
conda activate fvdb

# Install the package in editable mode (this sets up all imports properly)
cd fvdb-examples/point_transformer_v3
pip install -e .

# This installs the package and makes:
# - fvdb_extensions importable
# - pointcept importable (from external/pointcept/pointcept)
# No need for manual sys.path manipulation!
```



## Quick Test

```bash
# Download test data
python scripts/data/download_example_data.py

# Run inference
python scripts/test/minimal_inference.py \
  --data-path data/scannet_samples_small.json \
  --voxel-size 0.1 \
  --patch-size 1024 \
  --batch-size 1

# Compare results
python scripts/test/compute_difference.py --stats_path_1 data/scannet_samples_small_output.json --stats_path_2 data/scannet_samples_small_output_gt.json
```


## Project Structure

- `fvdb_extensions/models/point_transformer_v3m1_fvdb.py` - Pointcept backbone (`PT-v3fvdb`) wrapping `fvdb.nn.PointTransformerV3`
- `fvdb_extensions/configs/` - Pointcept training configs for the fvdb backbone and the official PT-v3m1
- `scripts/data/` - Data download and preprocessing scripts
- `scripts/test/` - Inference and comparison scripts

