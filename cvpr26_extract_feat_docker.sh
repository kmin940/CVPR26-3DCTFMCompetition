#/bin/bash

target='gallstone'

# python cvpr26_extract_feat_docker.py \
#     -i /home/jma/Documents/cryoSumin/CT_FM/data/raw_data_classify/amos-clf-tr-val/images \
#     -o ./results/${target} \
#     -d /home/jma/Documents/cryoSumin/CT_FM/CT-NEXUS \
#     -m /home/jma/Documents/cryoSumin/CT_FM/data/raw_data_classify/amos-clf-tr-val/fg_masks/${target} \

python cvpr26_extract_feat_docker.py \
    -i /home/jma/Documents/cryoSumin/CT_FM/data/raw_data_classify/amos-clf-tr-val/images \
    -o ./results/${target} \
    -d /home/jma/Documents/cryoSumin/CT_FM/CT-NEXUS


# python cvpr26_extract_feat_docker.py \
#     -i /path/to/AMOS-clf-tr-val/images \
#     -l /path/to/AMOS-clf-tr-val/labels \
#     -o ./results/${target} \
#     -d ./team_dockers \
#     -m /path/to/AMOS-clf-tr-val/fg_masks/${target} \