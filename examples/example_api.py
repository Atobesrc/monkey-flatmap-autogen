"""Use the package from Python: cut and flatten one hemisphere, inspect the gates.

Edit SUBJECTS_DIR / SUBJECT below.  The steps are the ones `mfa flatten` runs.
"""

import os

from mfa.flatten import canonical_orient, hemi_quality, passes_gates, quality_line, slim_flatten
from mfa.layout import flat_layout
from mfa.patch_io import write_patch
from mfa.recipe import default_recipe, route_kwargs
from mfa.surface import Hemi

SUBJECTS_DIR = os.environ.get("SUBJECTS_DIR", "<SUBJECTS_DIR>")
SUBJECT = "<SUBJECT>"
HEMI = "lh"
NAME = "example"  # scratch patch name: surf/<hemi>.example.{patch,flat.patch}.3d

sd = os.path.join(SUBJECTS_DIR, SUBJECT)
recipe = default_recipe()                      # the shared landmark route (+ default slit width)
H = Hemi(sd, HEMI)                             # white surface, cortex.label, parcellation
paths = H.seam_paths(("calcarine", "frontal", "temporal"), **route_kwargs(recipe))
print({k: len(p) for k, p in paths.items()}, H.seam_info["temporal"])

# narrowest slit width that passes every gate
for width in (1, 2, 3):
    ni, border, stats = H.build_slit_patch(paths, width)
    if stats["loops"] != 1 or stats["isolated"]:
        print(f"width {width}: patch is not a disk ({stats})")
        continue
    write_patch(f"{sd}/surf/{HEMI}.{NAME}.patch.3d", ni, H.v, border)
    slim_flatten(sd, HEMI, NAME)               # uniform Tutte + SLIM -> <hemi>.example.flat.patch.3d
    canonical_orient(sd, HEMI, NAME, backup=False)
    q = hemi_quality(sd, HEMI, NAME)
    print(f"width {width}: {quality_line(q)}")
    if passes_gates(q):
        break

layout = flat_layout(sd, HEMI, NAME)           # normalised parcel centroids, seam sides, outline
for parcel, rec in sorted(layout["parcels"].items()):
    print(f"{parcel:10s} centroid {rec['cent'].round(1)}  side {rec['side']:+d} (frac {rec['side_frac']:.2f})")
