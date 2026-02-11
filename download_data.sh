#!/usr/bin/env bash
source .venv/bin/activate

hf download kmin06/CVPR26-3DCTFMCompetition \
    --repo-type dataset \
    --local-dir . \
    --include "AMOS-clf-tr-val/*"