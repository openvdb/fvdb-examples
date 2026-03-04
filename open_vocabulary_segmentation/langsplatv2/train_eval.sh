#! /bin/bash
set -ex
for scene in ramen figurines teatime waldo_kitchen; do
  for level in 1 2 3; do
    python train_langsplatv2.py \
      --sfm-dataset-path data/lerf_ovs/${scene} \
      --reconstruction-path reconstructions/${scene}.ply \
      --config.feature-level $level \
      --run-name ${scene}_level_${level} \
      --log-path langsplatv2_logs \
      --config.max-steps 10000 \
      --preprocess.sam-model sam2
  done

  # Collect checkpoints (final_checkpoint.pt is saved at the run's top level)
  mkdir -p langsplatv2_results
  for level in 1 2 3; do
    cp langsplatv2_logs/${scene}_level_${level}/final_checkpoint.pt \
       langsplatv2_results/${scene}_level_${level}.pt
  done
done
