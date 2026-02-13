# LangSplatV2 (fVDB)

LangSplatV2-style open-vocabulary 3D segmentation using [fVDB](https://github.com/openvdb/fvdb-core) and pre-trained Gaussian splat reconstructions. This implementation trains per-Gaussian sparse coefficient fields and shared CLIP-aligned codebooks on an existing reconstruction; it does not train the underlying Gaussians or colors.

## What this implements

- **Preprocessing**: Multi-scale SAM2 masks and OpenCLIP feature encoding for each image (cached on disk).
- **Training**: Residual VQ codebooks and per-splat sparse logits so that rendered language features match the CLIP embeddings from SAM masks. One feature level (scale) per run; train multiple levels separately and combine at inference.
- **Compatibility**: Same feature pipeline and training setup (loss, LR, layer schedule) as the original LangSplatV2; uses fVDB for the 3D representation and rendering.

## Prerequisites

- **SfM scene**: COLMAP, `simple_directory`, or E57 dataset (images + cameras + optional point cloud).
- **Pre-trained Gaussian splat reconstruction**: A `.ply` or `.pt` / `.pth` checkpoint produced by e.g. [fvdb-reality-capture](https://github.com/openvdb/fvdb-reality-capture) or another fVDB-compatible pipeline. The script uses its normalization transform so the scene and Gaussians are aligned.

## Installation

From this directory (`open_vocabulary_segmentation/langsplatv2/`), with the `fvdb` conda environment active:

```bash
conda activate fvdb
pip install -e .
```

Dependencies (see `pyproject.toml`) include `torch`, `open-clip-torch`, `fvdb-reality-capture`, `tyro`, and optional TensorBoard for logging.

## How to run

Training loads the SfM scene, applies preprocessing (SAM2 + CLIP) with caching, then runs the language-feature training loop.

**Minimal (COLMAP scene + PLY reconstruction):**

```bash
python train_langsplatv2.py \
    --sfm-dataset-path /path/to/colmap/scene \
    --reconstruction-path /path/to/point_cloud.ply
```

**With explicit feature level and log directory:**

```bash
python train_langsplatv2.py \
    --sfm-dataset-path /path/to/colmap/scene \
    --reconstruction-path /path/to/point_cloud.ply \
    --config.feature-level 1 \
    --log-path langsplatv2_logs
```

**Train all three scale levels (as in the paper):**

```bash
for level in 1 2 3; do
  python train_langsplatv2.py \
    --sfm-dataset-path /path/to/scene \
    --reconstruction-path /path/to/gaussians.ply \
    --config.feature-level $level \
    --log-path langsplatv2_logs
done
```


**Useful flags:**

- `--config.feature-level` — 0=default, 1=small, 2=medium, 3=large (default: 1).
- `--config.max-steps` — Training steps (default from max_epochs if not set).
- `--preprocess.image-downsample-factor` — Downsample images before SAM2/CLIP (e.g. 2 for speed).
- `--preprocess.sam2.checkpoint` — SAM2 size: `large`, `small`, `tiny`, `base_plus`.
- `--log-path` — Directory for run subdirs (checkpoints, metrics). Use `None` to disable saving.
- `--io.use-tensorboard` — Log scalars (and optionally images) to TensorBoard.
- `--use-every-n-as-val` — Hold out every N-th image for validation (e.g. 5); -1 = no validation.

## Outputs

With `--log-path` set (e.g. `langsplatv2_logs`), each run writes:

- `log_path/run_<timestamp>/` (or `log_path/<run_name>/` if `--run-name` is set)
  - `checkpoints/<step>/langsplatv2_ckpt.pt` — Model state and config (when `io.save_checkpoints` is True).
  - `metrics_log.csv` — Step, loss, and optional validation metrics.
  - `tensorboard/` — If `io.use_tensorboard` is True.
  - `images/` — If `io.save_images` is True (e.g. feature visualizations at save steps).

Preprocessing caches (SAM2 masks, CLIP features) are stored under the scene’s cache directory and reused across runs.

## Preprocessing pipeline and cache format

The pipeline (see `LangSplatV2PreprocessConfig` in `config.py`) runs in order: optional scene normalization, point filtering, image downsampling, filter images by visible points, **ComputeMultiScaleSAM2Masks**, **ComputeCLIPFeatures**, and optional cropping.

### SAM2 masks (per image)

- `{scale}_segmentations`: `[N, H, W]` binary masks
- `{scale}_bboxes`: `[N, 4]` XYWH
- `{scale}_areas`, `{scale}_predicted_ious`, `{scale}_stability_scores`

Scales: `default`, `s` (small, &lt;1% area), `m` (1–10%), `l` (≥10%).

### CLIP features (per image)

- `features`: `[N_total, 512]` L2-normalized OpenCLIP embeddings (one per mask, concatenated over scales).
- `seg_maps`: `[4, H, W]` — pixel → feature index per scale (-1 = no mask).
- `lengths`: `[4]` — number of masks per scale (default, s, m, l).

Training uses a single `feature_level` (0–3) to choose which scale’s seg map and features to use.

## Training details and comparison with original LangSplatV2

- **Feature generation**: Same as original — crop mask region → pad to square → resize to 224 → OpenCLIP encode → L2-normalize. Scale order and seg-map indexing (default → s → m → l, cumulative) match.
- **Optimization**: Same language-feature LR (0.0025), layer schedule (every 10k steps), and cosine loss over valid pixels with gradient scaling via mask fraction. The scalar `train/loss` is the (mask-fraction-scaled) total loss used for backprop. For a smoother, more interpretable curve when mask coverage varies across images, use `train/cosine_loss_valid`, which is the mean cosine loss over valid pixels only (no mask-fraction scaling), we use this for logging.
- **Data sampling**: One random permutation of all training views per “epoch” (InfiniteSampler with shuffle), one view per step when `batch_size=1`, matching the original’s viewpoint-stack behavior.

## References

- [LangSplatV2: Vision-Language Gaussian Splatting](https://arxiv.org/abs/2312.16084)
- [Segment Anything 2 (SAM2)](https://github.com/facebookresearch/segment-anything-2)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
