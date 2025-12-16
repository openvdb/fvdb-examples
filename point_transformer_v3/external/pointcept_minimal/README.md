# Pointcept Minimal

A minimal version of [Pointcept](https://github.com/Pointcept/Pointcept) for comparing official PTv3 and fVDB PTv3 training results on ScanNet.

## Environment Setup

1. **fVDB-core**: Follow fVDB's installation instructions and install `env/dev_environment.yml`. Tested with fVDB commit `51cec3d3e90d7d571e22862d17ae9051cbd13afd`.
2. Activate the environment and install dependencies:
   ```bash
   conda activate fvdb
   pip install -r requirements_pointceptminimal.txt
   ```

## Dataset Preparation

Follow the [Pointcept scannet dataset preparation](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#scannet-v2) to download ScanNet and prepare the dataset in the correct location.

## Training Configurations

### With Convolution (CPE)
Both configurations train PTv3 on ScanNet with convolution enabled. Training loss curves will diverge. 

**fVDB PTv3:**
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 sh scripts/train.sh -g 4 -d scannet -c semseg-pt-v3m1-0-fvdb-test-4g-conv -n semseg-pt-v3m1-0-fvdb-test-4g-conv
```

**Official PTv3:**
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 sh scripts/train.sh -g 4 -d scannet -c semseg-pt-v3m1-0-test-4g-conv -n semseg-pt-v3m1-0-test-4g-conv
```

### Without Convolution (CPE)
Both configurations train PTv3 on ScanNet without convolution. Training loss curves are identical. 

**fVDB PTv3:**
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 sh scripts/train.sh -g 4 -d scannet -c semseg-pt-v3m1-0-fvdb-test-4g -n semseg-pt-v3m1-0-fvdb-test-4g
```

**Official PTv3:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/train.sh -g 4 -d scannet -c semseg-pt-v3m1-0-test-4g -n semseg-pt-v3m1-0-test-4g
```

## Weights & Biases (Optional)

To enable W&B logging, set the following in your config file:
```python
enable_wandb = True
wandb_project = "your_project"
wandb_key = "your_key"
```
