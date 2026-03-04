#! /bin/bash
export PYTHONUNBUFFERED=1

for scene in  teatime waldo_kitchen; do
  frgs reconstruct \
    --run-name ${scene} \
    --tx.image-downsample-factor 1 \
    data/lerf_ovs/${scene}/ \
    -uv 10 \
    -o reconstructions/${scene}.ply \
    --cfg.batch-size 1 \
    --cfg.pose_opt_start_epoch 20
done
