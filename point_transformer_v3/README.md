# Point Transformer V3 (PT-v3) - FVDB Implementation

This repository contains a minimal implementation of Point Transformer V3 using the FVDB library for scalable 3D point cloud processing.

## Environment

Use the FVDB default development environment:

```bash
conda env create -f env/dev_environment.yml
```

Next, activate the environment and install additional dependancies specifically for the point transformer project.

```bash
pip install -r requirements.txt
```


## Files Overview

### 2. `prepare_scannet_dataset.py`
**Purpose**: Prepares ScanNet dataset samples for testing and development

**Prerequisites**:
- Download the full ScanNet dataset from https://github.com/ScanNet/ScanNet (requires application approval)
- Store the dataset to a local directory (e.g., `/path/to/scannet`)

**Usage**:
```bash
python prepare_scannet_dataset.py --data_root /path/to/scannet --output_file scannet_samples.json --num_samples 10
```

**What it does**:
- Loads ScanNet dataset from specified root directory where it is downloaded
- Performs grid sampling to reduce point density
- Exports a subset of samples to a single JSON file: the `scannet_samples.json` containing point coordinates, colors, and labels

### 1. `download_example_data.py`
**Purpose**: Download the preprocessed ScanNet dataset samples for testing.

**Usage**:
```bash
python download_example_data.py
```

**What it does**:
- Downloads a pre-processed ScanNet sample set together with the corresponding PT-v3 reference outputs. This replicates the result of running `python prepare_scannet_dataset.py` locally, but saves you from downloading the entire ScanNet dataset and performing the preprocessing yourself.
- The script provides a single set of samples; to generate additional datasets, run `python prepare_scannet_dataset.py` instead.


### 2. `model.py`
**Purpose**: Implements the PT-v3 architecture using FVDB.

**Key Components**:
- `PTV3`: Main model class with configurable encoder depths and channels
- `PTV3_Encoder`: A PT-v3 encoder consisting of multiple PT-v3 block. The grid resolution remained the same throughout the encoder
- `PTV3_Block`: Transformer block with attention and MLP
- `PTV3_CPE`: Convolutional Positional Encoding
- `PTV3_Attention`: Multi-head self-attention
- `PTV3_Pooling`: Downsampling operations

**Usage**: Imported by `minimal_inference.py` for model instantiation.

### 3. `minimal_inference.py`
**Purpose**: Demonstrates PT-v3 inference on ScanNet point clouds.

**Usage**:
```bash
python minimal_inference.py
```

**What it does**:
- Loads point cloud data from `scannet_samples.json`
- Converts ScanNet data to fVDB format
- Runs PT-v3 model inference
- Saves runtime statistics to `runtime_stats.json`

**Prerequisites**: Requires `scannet_samples.json` from `prepare_scannet_dataset.py`

### 4. `compute_difference.py`
**Purpose**: Compares inference results between fVDB implementation and original PT-v3 implementation.

**Usage**:
```bash
python compute_difference.py --stats_path_1 stats1.json --stats_path_2 stats2.json
```

**What it does**:
- Loads two `runtime_stats.json` files
- Computes average absolute and relative deviations
- Reports differences in output features, sums, and last elements
- Useful for validating model changes or comparing implementations

## Test PT-v3

To test the Point Transformer V3 implementation, follow these steps:

### Step 1: Download the Dataset

First, download the preprocessed ScanNet dataset samples and reference outputs:

```bash
python download_example_data.py
```

This will download the following files to the `data/` directory:
- `scannet_samples_small.json` - Small point-clouds, each has a few thousands points.
- `scannet_samples_large.json` - Larger point-clouds, each has 50k~100k points.
- `scannet_samples_small_output_gt.json` - Reference outputs for small dataset.
- `scannet_samples_large_output_gt.json` - Reference outputs for large dataset.

### Step 2: Inference point transformer PT-v3

Run the PT-v3 model inference on the downloaded samples:

```bash
# Test with small dataset
python minimal_inference.py --data-path data/scannet_samples_small.json --voxel-size 0.1 --patch-size 1024

# Test with large dataset
python minimal_inference.py --data-path data/scannet_samples_large.json --voxel-size 0.02 --patch-size 1024
```

This will:
- Load the point cloud data from the JSON file
- Convert the data to fVDB format
- Run PT-v3 model inference
- Save runtime statistics and results to the specified output file

### Step 3: Compute the Difference

Compare your inference results with the reference outputs to validate the implementation:

```bash
# Compare small dataset results
python compute_difference.py --stats_path_1 data/scannet_samples_small_output.json --stats_path_2 data/scannet_samples_small_output_gt.json

# Compare large dataset results
python compute_difference.py --stats_path_1 data/scannet_samples_large_output.json --stats_path_2 data/scannet_samples_large_output_gt.json
```

This will:
- Load both result files (your inference results and reference outputs)
- Compute average absolute and relative deviations
- Report differences in output features, sums, and last elements
- Expect only small numerical differences (typically < 1e-5) due to floating-point precision.

