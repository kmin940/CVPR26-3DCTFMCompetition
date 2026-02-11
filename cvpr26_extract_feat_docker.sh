#/bin/bash

python cvpr26_extract_feat_docker.py \
    -i /home/jma/Documents/cryoSumin/CT_FM/data/raw_data_classify/amos-clf-tr-val/images \
    -l /home/jma/Documents/cryoSumin/CT_FM/data/raw_data_classify/amos-clf-tr-val/labels \
    -o ./results \
    -d /home/jma/Documents/cryoSumin/CT_FM/CT-NEXUS \
    -m /home/jma/Documents/cryoSumin/CT_FM/data/raw_data_classify/amos-clf-tr-val/fg_masks \
    --target fatty_liver

# python cvpr26_extract_feat_docker.py \
#     -i /path/to/amos-clf-tr-val/images \
#     -l /path/to/amos-clf-tr-val/labels \
#     -o ./results \
#     -d ./team_dockers \
#     --target splenomegaly \
#     --keep_temp