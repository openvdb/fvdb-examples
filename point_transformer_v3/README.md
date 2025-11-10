# Point Transformer V3 (PT-v3) - FVDB Implementation

This repository contains a minimal implementation of Point Transformer V3 using the FVDB library for scalable 3D point cloud processing.

## Environment

Use the FVDB default development environment and install FVDB package:

```bash
cd fvdb/
conda env create -f env/dev_environment.yml
conda activate fvdb
./build.sh
```

Next, activate the environment and install additional dependancies specifically for the point transformer project.

```bash
cd fvdb-examples/point_transformer_v3
pip install -r requirements.txt
```

In order to train on Scannet dataset with pointcept codebase, we need to additionally install:

```bash
cd fvdb-examples/point_transformer_v3
pip install -r requirements_pointcept.txt
```



## Files Overview

### 2. `prepare_scannet_dataset.py`
**Purpose**: Prepares ScanNet dataset samples for testing and development

**Prerequisites**:
- Download the full ScanNet dataset from https://github.com/ScanNet/ScanNet (requires application approval)
- Store the dataset to a local directory (e.g., `/path/to/scannet`)

**Usage**:
```bash
python prepare_scannet_dataset.py --data_root /path/to/scannet --output_file scannet_samples.json --num_samples 16
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
python minimal_inference.py --data-path data/scannet_samples_small.json --voxel-size 0.1 --patch-size 1024 --batch-size 1

# Test with large dataset
python minimal_inference.py --data-path data/scannet_samples_large.json --voxel-size 0.02 --patch-size 1024 --batch-size 1
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

## Training on ScanNet Dataset

This section describes how to train PT-v3 models on the ScanNet dataset using the minimal Pointcept training codebase, using either their ptv3 implementation and our fVDB implementation. 

### Environment Setup

Follow the **Environment** section above to set up the development environment and install all required dependencies.

### ScanNet Dataset Preparation

The preprocessing pipeline supports semantic and instance segmentation for `ScanNet20`, `ScanNet200`, and `ScanNet Data Efficient` benchmarks.

1. **Download the dataset**: Obtain the [ScanNet v2 dataset](http://www.scan-net.org/) (requires registration and approval).

2. **Preprocess the raw data**: Run the preprocessing script to convert the raw ScanNet data into the required format:

```bash
# RAW_SCANNET_DIR: the directory containing the downloaded ScanNet v2 raw dataset
# PROCESSED_SCANNET_DIR: the output directory for the processed ScanNet dataset
python pointcept_minimal/pointcept/datasets/preprocessing/scannet/preprocess_scannet.py \
    --dataset_root ${RAW_SCANNET_DIR} \
    --output_root ${PROCESSED_SCANNET_DIR}
```

3. **Alternative**: Download preprocessed data directly from [HuggingFace](https://huggingface.co/datasets/Pointcept/scannet-compressed). Please ensure you agree to the official ScanNet license before downloading.

4. **Link the processed dataset** to the codebase data directory:

```bash
# PROCESSED_SCANNET_DIR: the directory containing the processed ScanNet dataset
mkdir -p pointcept_minimal/data
ln -s ${PROCESSED_SCANNET_DIR} pointcept_minimal/data/scannet
```

### Training Scripts

To train the PT-v3 models with different configurations, use the following commands from the `pointcept_minimal` directory:

```bash
# Train PT-v3 with FVDB backend (8 GPUs)
sh scripts/train.sh -g 8 -d scannet -c semseg-pt-v3m1-0-fvdb-test -n semseg-pt-v3m1-0-fvdb-test

# Train PT-v3 with standard backend (8 GPUs)
sh scripts/train.sh -g 8 -d scannet -c semseg-pt-v3m1-0-test -n semseg-pt-v3m1-0-test
```

You should launch the above scripts within `point_transformer_v3/pointcept_minimal` folder. 

### Configuration Files

The training configurations are located at:
- `pointcept_minimal/configs/scannet/semseg-pt-v3m1-0-fvdb-test.py` - FVDB-based implementation
- `pointcept_minimal/configs/scannet/semseg-pt-v3m1-0-test.py` - Standard implementation

### Model Implementation

The model implementations can be found in the following files:
- `pointcept_minimal/pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py` - Base PT-v3 implementation
- `pointcept_minimal/pointcept/models/point_transformer_v3/point_transformer_v3m1_fvdb.py` - FVDB-accelerated PT-v3 implementation

