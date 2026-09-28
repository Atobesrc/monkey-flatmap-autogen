#!/usr/bin/env bash
# Flatten a new FreeSurfer subject with the shared cut recipe and import it into pycortex.
#
# Placeholders: <SUBJECTS_DIR> (FreeSurfer subjects directory), <SUBJECT> (folder name in it,
# e.g. sub-01), <CX> (pycortex subject name), <OUT> (output folder for figures and records).
# Requirements: the conda/venv with this package (and pycortex for steps 3-5), Inkscape on
# PATH or configured in ~/.config/pycortex/options.cfg for the overlay rendering.
set -euo pipefail

export SUBJECTS_DIR=<SUBJECTS_DIR>
SUBJECT=<SUBJECT>
CX=<CX>
OUT=<OUT>
# fixed BLAS/OpenMP thread count: the SLIM result is bit-reproducible only under a fixed count
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

# 1. cut + flatten both hemispheres with the packaged recipe; narrowest passing slit width per hemisphere
mfa flatten --subject "$SUBJECT" --hemis lh rh --name uc --slit-width auto --out-dir "$OUT"

# 2. look at the gates and figures before promoting
cat "$OUT/${SUBJECT}_uc/chosen.json" | head -40
ls "$OUT/${SUBJECT}_uc/"*_qc.png

# 3. promote to <hemi>.flatten.*, create/update the pycortex subject, regenerate overlays.svg (backups made)
mfa apply --subject "$SUBJECT" --name uc --pycortex-subject "$CX" --out-dir "$OUT"

# 4. parcel outlines + labels into the 'rois' layer of overlays.svg, and a rendered figure
mfa overlay-parcels --subject "$SUBJECT" --pycortex-subject "$CX" \
    --render "$OUT/${SUBJECT}_uc/flatmap_parcels.png" --render-title ""

# 5. (optional) another parcellation: resample a label volume, then draw it as its own layer
# mfa atlas-annot --subject "$SUBJECT" --pycortex-subject "$CX" --volume <ATLAS.nii.gz> --lookup <ATLAS.tsv> \
#     --out-name aparc.LEVEL3.mapped --extra-valid HF pAmy spAmy
# mfa overlay-parcels --subject "$SUBJECT" --pycortex-subject "$CX" --annot aparc.LEVEL3.mapped --layer LEVEL3 \
#     --render "$OUT/${SUBJECT}_uc/flatmap_level3.png"
