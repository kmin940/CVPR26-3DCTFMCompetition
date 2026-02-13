#/bin/bash

target='gallstone'

python cvpr26_extract_feat_docker.py \
    -i /path/to/AMOS-clf-tr-val/images \
    -o ./results/${target} \
    -d /path/to/docker
