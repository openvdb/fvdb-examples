# GARƒVDB

Scale-conditioned segmentation for Gaussian splatting using ƒVDB.

## Setup Environment

```bash
conda env create -f ./garfvdb_environment.yml
conda activate fvdb_garfvdb
```

### Build and Install ƒVDB

```bash
# Clone and build ƒVDB
git clone https://github.com/openvdb/fvdb-core.git
cd fvdb-core
conda env create -f env/build_environment.yml
conda activate fvdb_build
./build.sh wheel verbose

# Install in garfvdb environment
conda activate fvdb_garfvdb
pip install ./dist/<fvdb_wheel>.whl

# Install ƒVDB Reality Capture
cd ..
git clone https://github.com/openvdb/fvdb-reality-capture.git
cd fvdb-reality-capture
pip install .
```

## Quick Start

This example uses the Safety Park dataset from ƒVDB Reality Capture.

### 1. Download Dataset and Train Gaussian Splats

```bash
# Download the dataset
frgs download safety_park

# Train 3D Gaussian splatting
frgs reconstruct \
    --run-name safety_park_1 \
    --tx.image-downsample-factor 2 \
    data/safety_park/
```

### 2. Train GARƒVDB Segmentation

```bash
python train_segmentation.py \
    --sfm-dataset-path data/safety_park/ \
    --reconstruction-path frgs_logs/safety_park_1/checkpoints/00024800/reconstruct_ckpt.pt \
    --use-every-n-as-val 80 \
    --io.use-tensorboard \
    --io.save-images-to-tensorboard \
    --io.save-images \
    --config.max-epochs 300 \
    --tx.image-downsample-factor 2
```

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir ./garfvdb_logs
# Open http://localhost:6006/
```

[In Development] Monitor training progress with the 3D viewer:

Add the following argument to the `train_segmentation.py` command to enable mask visualization updates every 10 epochs:

```bash
--visualize-every 10
# Open http://localhost:8080/ to view the training progress in the 3D NanoVDB Editor Visualization.
```



### 3. [In Development] Visualize Results

Launch the interactive viewer:

```bash
python view_checkpoint.py \
    --reconstruction-path frgs_logs/safety_park_1/checkpoints/00024800/reconstruct_ckpt.pt \
    --segmentation-path garfvdb_logs/<run_name>/checkpoints/00036600/train_ckpt.pt \
    --initial-blend 0.2 \
    --initial-scale 0.1
# Open http://localhost:8080/ to view the results.
```


**Viewer Controls:**
- `--initial-blend`: Mask opacity (0 = transparent, 1 = opaque)
- `--initial-scale`: Segmentation scale as fraction of max scene scale (0-1)
