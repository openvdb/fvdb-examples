# LERF-OVS Evaluation

Open-vocabulary segmentation evaluation on the [LERF-OVS](https://www.lerf.io/) dataset, comparing against ground-truth labelme annotations. Computes segmentation mIoU and localization accuracy across four scenes: `ramen`, `figurines`, `teatime`, `waldo_kitchen`.

All commands below should be run from this directory (`evaluation/lerf_ovs/`).

## Prerequisites

- The `fvdb` conda environment with the `fvdb-langsplatv2` package installed (see the [parent README](../../README.md)).
- `fvdb-reality-capture` installed.

## Step 1: Download the LERF-OVS dataset

```bash
python download_data.py --dataset-root data
```

This downloads and extracts the LERF-OVS data from Google Drive into `data/lerf_ovs/`. The resulting layout is:

```
data/lerf_ovs/
    label/<scene>/frame_XXXXX.json, frame_XXXXX.jpg
    <scene>/images/, sparse/
```

## Step 2: Reconstruct Gaussian splats

```bash
bash batch_reconstruct_eval_scenes.sh
```

This runs `frgs reconstruct` on each scene with default settings. Outputs are written to:

```
reconstructions/<scene>.ply
```

To reconstruct a single scene manually:

```bash
frgs reconstruct \
    --run-name teatime \
    --tx.image-downsample-factor 1 \
    data/lerf_ovs/teatime/ \
    -uv 10 \
    -o reconstructions/teatime.ply \
    --cfg.batch-size 1 \
    --cfg.pose_opt_start_epoch 20
```

## Step 3: Train LangSplatV2 features

```bash
bash batch_train_eval_langsplat.sh
```

For each scene, this trains three models (one per SAM scale level: 1=small, 2=medium, 3=large) for 10k steps using SAM2. The final checkpoints are collected into:

```
langsplatv2_results/<scene>_level_1.pt
langsplatv2_results/<scene>_level_2.pt
langsplatv2_results/<scene>_level_3.pt
```

To train a single scene and level manually:

```bash
python ../../scripts/train_langsplatv2.py \
    --sfm-dataset-path data/lerf_ovs/teatime \
    --reconstruction-path reconstructions/teatime.ply \
    --config.feature-level 1 \
    --run-name teatime_level_1 \
    --log-path langsplatv2_logs \
    --config.max-steps 10000 \
    --preprocess.sam-model sam2
```

Then copy the final checkpoint:

```bash
cp langsplatv2_logs/teatime_level_1/final_checkpoint.pt \
   langsplatv2_results/teatime_level_1.pt
```

## Step 4: Evaluate

**All scenes (auto-discovered from checkpoints):**

```bash
python eval_lerf.py \
    --lerf-root data/lerf_ovs \
    --results-root langsplatv2_results \
    --reconstructions-root reconstructions
```

**Single scene:**

```bash
python eval_lerf.py \
    --lerf-root data/lerf_ovs \
    --results-root langsplatv2_results \
    --reconstructions-root reconstructions \
    --scenes teatime
```

The evaluation:
1. Loads all three level checkpoints per scene
2. Renders CLIP features from each level for each annotated frame
3. Computes OpenCLIP relevancy maps for each ground-truth text prompt
4. Selects the best level per prompt (highest max relevancy score)
5. Reports **segmentation mIoU** (thresholded relevancy vs GT masks) and **localization accuracy** (relevancy peak inside GT bounding box)

### Evaluation flags

- `--mask-thresh` — Relevancy threshold for binary segmentation mask (default: 0.4).
- `--eval-topk` — Number of codebook entries to combine at eval (default: 4).
- `--output-dir` — Where to write results and visualizations (default: `lerf_eval_results`).
- `--no-visualizations` — Skip saving per-frame visualization images.
- `--verbose` — Enable debug logging.

### Output

Results are saved to `lerf_eval_results/` (or the path given by `--output-dir`):

```
lerf_eval_results/
    lerf_results.json              # Summary across all scenes (mIoU, localization accuracy)
    <scene>/
        results.json               # Per-frame breakdown for this scene
        frame_XXXXX.jpg            # Per-frame visualizations (if enabled)
```
