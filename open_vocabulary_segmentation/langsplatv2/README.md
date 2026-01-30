# LangSplatV2

This project implements LangSplatV2 (Li, et al. 2025) with fVDB for open-vocabulary 3D segmentation.

## Overview

The LangSplatV2 scene data transformation pipeline consists of two main steps:

1. **Multi-scale SAM2 Segmentation**: Uses SAM2 to generate segmentation masks at multiple scales (default, small, medium, large) for each image.

2. **CLIP Feature Encoding**: Encodes each segmented region using OpenCLIP to produce language-aligned features that can be used for open-vocabulary queries.

## Installation

```bash
# Install from the fvdb-examples repository
pip install -e .

# Or install dependencies manually
pip install open-clip-torch fvdb-reality-capture
```


## Scene Transform Outputs

### SAM2 Masks

For each image, the SAM2 transform produces:

- `{scale}_segmentations`: Binary masks, shape `[N, H, W]`
- `{scale}_bboxes`: Bounding boxes in XYWH format, shape `[N, 4]`
- `{scale}_areas`: Mask areas in pixels, shape `[N]`
- `{scale}_predicted_ious`: SAM2's IoU predictions, shape `[N]`
- `{scale}_stability_scores`: Mask stability scores, shape `[N]`

where `{scale}` is one of: `default`, `s` (small), `m` (medium), `l` (large).

Masks are categorized by area ratio:
- **Large (l)**: area >= 10% of image
- **Medium (m)**: 1% <= area < 10%
- **Small (s)**: area < 1%
- **Default**: all masks

### CLIP Features

For each image, the CLIP transform produces:

- `features`: CLIP embeddings, shape `[N_total, 512]`
- `seg_maps`: Segmentation maps, shape `[4, H, W]`
- `lengths`: Number of masks per scale, shape `[4]`

The `seg_maps` tensor maps each pixel to a feature index (or -1 for unmasked pixels).

## References

- [LangSplatV2: Vision-Language Gaussian Splatting](https://arxiv.org/abs/2312.16084)
- [Segment Anything 2 (SAM2)](https://github.com/facebookresearch/segment-anything-2)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
