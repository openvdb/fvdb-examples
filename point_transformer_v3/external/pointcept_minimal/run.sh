
# sh scripts/train.sh -g 8 -d scannet -c semseg-pt-v3m1-0-test-fw -n semseg-pt-v3m1-0-test-fw

# sh scripts/train.sh -g 8 -d scannet -c semseg-pt-v3m1-0-fvdb-test-fw -n semseg-pt-v3m1-0-fvdb-test-fw

# sh scripts/train.sh -g 8 -d scannet -c semseg-pt-v3m1-0-test -n semseg-pt-v3m1-0-test

# sh scripts/train.sh -g 8 -d scannet -c semseg-pt-v3m1-0-fvdb-test -n semseg-pt-v3m1-0-fvdb-test

CUDA_VISIBLE_DEVICES=0 sh scripts/train.sh -g 1 -d scannet -c semseg-pt-v3m1-0-test-1g -n semseg-pt-v3m1-0-test-1g

CUDA_VISIBLE_DEVICES=1 sh scripts/train.sh -g 1 -d scannet -c semseg-pt-v3m1-0-fvdb-test-1g -n semseg-pt-v3m1-0-fvdb-test-1g

CUDA_VISIBLE_DEVICES=2 sh scripts/train.sh -g 1 -d scannet -c semseg-pt-v3m1-0-test-1g-2 -n semseg-pt-v3m1-0-test-1g-2

CUDA_VISIBLE_DEVICES=3 sh scripts/train.sh -g 1 -d scannet -c semseg-pt-v3m1-0-fvdb-test-1g-2 -n semseg-pt-v3m1-0-fvdb-test-1g-2
