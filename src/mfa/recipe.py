"""Cut recipes: one set of landmark rules shared by every hemisphere of every subject.

A recipe is a dictionary of six fields (:data:`RECIPE_FIELDS`).  Five of them
define *where* the cuts run, using only parcel membership and anterior/dorsal
ordering, so they are mirror-symmetric and subject-independent by construction:

``temporal_src``
    rim entry of the temporal seam: ``nearest_<landmark>`` = the medial-wall rim
    vertex nearest an anatomical landmark (``<P>_ant``, ``<P>_post``,
    ``<A>_<B>_border`` or ``<P1>_<P2>.._centroid``).
``temporal_dst``
    destination of the temporal seam: ``<P>_ant`` (most anterior vertex of parcel
    P) or ``<A>_<B>_border`` (most anterior vertex on the A/B border).
``temporal_corridor``
    parcels the temporal seam may run through: ``medial`` (MTL, Amy, TG, ITC),
    ``lateral`` (TG, ITC only) or ``border`` (medial corridor with edges inside
    Amy/MTL penalised by :data:`BORDER_PENALTY`).
``temporal_extend``
    optional posterior continuation of the seam inside TG/ITC to a landmark
    (:data:`TEMPORAL_EXTEND_OPTIONS`); ``none`` = no extension.
``calcarine_extend``
    optional continuation of the calcarine cut within V1/V2-V4
    (:data:`CALCARINE_EXTEND_OPTIONS`); ``none`` = stop at the most posterior V1 vertex.

The sixth field, ``slit_width`` (vertex rings removed around the route), is a
technical mesh parameter and not part of the anatomical standard: the recipe
value is only a default, and the standard is the narrowest width that passes
every quality gate on each hemisphere (``--slit-width auto``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from importlib import resources

TEMPORAL_CORRIDORS = ("medial", "lateral", "border")
BORDER_PENALTY = 4.0  # 'border' corridor: edge cost multiplier inside Amy/MTL

#: Parcels whose bank of the temporal seam is compared between hemispheres / subjects.
SEAM_PARCELS = ("Amy", "MTL", "MPal", "TG", "ITC")
#: Parcels the temporal slit may run through (identifies the temporal slit of an existing patch).
TEMPORAL_CORRIDOR_PARCELS = ("MTL", "Amy", "TG", "ITC")
#: Spellings usable in landmark rules for parcels whose name contains '-'.
PARCEL_ALIASES = {"V2V4": "V2-V4"}

TEMPORAL_SRC_OPTIONS = (
    "nearest_MTL_Amy_centroid",  # rim vertex nearest the MTL+Amy centroid
    "nearest_Amy_ant",           # rim vertex nearest the most anterior Amy vertex
    "nearest_Amy_MPal_border",   # ... nearest the Amy/MPal border (its centroid)
    "nearest_MPal_MTL_border",   # ... nearest the MPal/MTL border
    "nearest_TG_MTL_border",     # ... nearest the TG/MTL border
)
TEMPORAL_DST_OPTIONS = (
    "TG_ant",             # most anterior TG vertex (temporal pole)
    "TG_ITC_border",      # most anterior TG/ITC border vertex
    "Amy_ant",            # most anterior Amy vertex
    "MTL_ant",            # most anterior MTL vertex
    "TG_STG/STSd_border",  # most anterior TG / STG-STSd border vertex
)
TEMPORAL_EXTEND_OPTIONS = (
    "none",           # no posterior extension
    "TG_post",        # along TG|ITC to the most posterior TG vertex
    "ITC_frac_0.25",  # to the ITC ventral midline at 25 % of its A-P extent
    "ITC_frac_0.5",   # ... at 50 %
    "MT_ant",         # to the ITC vertex nearest the anterior tip of MT
    "V2V4_border",    # posterior along ITC until the ITC / V2-V4 border
)
CALCARINE_EXTEND_OPTIONS = (
    "none",       # stop at the most posterior V1 vertex
    "V2V4_post",  # continue within V1|V2-V4 to the most posterior V2-V4 vertex
)
RECIPE_FIELDS = ("slit_width", "temporal_src", "temporal_dst", "temporal_corridor",
                 "temporal_extend", "calcarine_extend")
ROUTE_FIELDS = RECIPE_FIELDS[1:]
RECIPE_FORMAT = "cut_recipe/1"
#: Slit widths tried, in order, by the ``auto`` width policy.
AUTO_SLIT_WIDTHS = (1, 2, 3)

# short codes used in scratch patch names (<base>_w2_Aa_TGa_lat_e0)
RECIPE_CODES = {
    "temporal_src": {"nearest_MTL_Amy_centroid": "MAc", "nearest_Amy_ant": "Aa",
                     "nearest_Amy_MPal_border": "AMb", "nearest_MPal_MTL_border": "PMb",
                     "nearest_TG_MTL_border": "TMb"},
    "temporal_dst": {"TG_ant": "TGa", "TG_ITC_border": "TIb", "Amy_ant": "Ama",
                     "MTL_ant": "MTa", "TG_STG/STSd_border": "TSb"},
    "temporal_corridor": {"medial": "med", "lateral": "lat", "border": "bor"},
    "temporal_extend": {"none": "e0", "TG_post": "eTGp", "ITC_frac_0.25": "eI25",
                        "ITC_frac_0.5": "eI50", "MT_ant": "eMTa", "V2V4_border": "eV4b"},
    "calcarine_extend": {"none": "", "V2V4_post": "kV4p"},
}

Recipe = dict[str, object]


def packaged_recipe_path() -> str:
    """Path of the recipe shipped with the package (``mfa/data/cut_recipe.json``)."""
    return str(resources.files("mfa").joinpath("data", "cut_recipe.json"))


def default_recipe() -> Recipe:
    """The packaged default recipe as a validated dictionary."""
    return load_recipe_file(packaged_recipe_path())


def _valid_landmark(spec: str, suffixes: tuple[str, ...]) -> bool:
    return any(spec.endswith(s) for s in suffixes)


def validate_recipe(r: Mapping[str, object]) -> Recipe:
    """Type-check and normalise a recipe mapping.

    Raises
    ------
    ValueError
        On a missing field, an unknown option or an unsupported legacy value
        (millimetre extensions ``mm:<x>`` and rim-entry shifts are not part of
        the landmark-only standard).
    """
    out: Recipe = dict(r)
    for k in RECIPE_FIELDS:
        if k not in out:
            raise ValueError(f"recipe: missing field {k!r} (fields: {RECIPE_FIELDS})")
    out["slit_width"] = int(out["slit_width"])  # type: ignore[arg-type]
    if out["slit_width"] < 1:  # type: ignore[operator]
        raise ValueError("recipe: slit_width must be >= 1")
    if out["temporal_corridor"] not in TEMPORAL_CORRIDORS:
        raise ValueError(f"recipe: temporal_corridor {out['temporal_corridor']!r} not in {TEMPORAL_CORRIDORS}")
    s = str(out["temporal_src"])
    if not (s.startswith("nearest_") and _valid_landmark(s, ("_centroid", "_border", "_ant", "_post"))):
        raise ValueError(f"recipe: temporal_src {s!r}: 'nearest_<landmark>' with landmark "
                         "'<P>_ant' | '<P>_post' | '<A>_<B>_border' | '<P1>_<P2>.._centroid'")
    d = str(out["temporal_dst"])
    if not _valid_landmark(d, ("_border", "_ant")):
        raise ValueError(f"recipe: temporal_dst {d!r}: '<P>_ant' or '<A>_<B>_border'")
    for field, options in (("temporal_extend", TEMPORAL_EXTEND_OPTIONS),
                           ("calcarine_extend", CALCARINE_EXTEND_OPTIONS)):
        e = str(out[field])
        if e.startswith("mm:"):
            raise ValueError(f"recipe: {field} {e!r}: millimetre extensions are not landmark rules "
                             "and are not supported; use one of "
                             f"{options} or a landmark ('<P>_ant' | '<P>_post' | '<A>_<B>_border')")
        if not (e in options or e.startswith("ITC_frac_") or _valid_landmark(e, ("_ant", "_post", "_border"))):
            raise ValueError(f"recipe: {field} {e!r} not in {options}")
        out[field] = e
    shift = float(out.get("temporal_src_shift_mm", 0.0) or 0.0)  # type: ignore[arg-type]
    if shift != 0.0:
        raise ValueError("recipe: temporal_src_shift_mm is not a landmark rule and is not supported")
    return {k: out[k] for k in RECIPE_FIELDS}


def load_recipe_file(fn: str) -> Recipe:
    """Load a recipe JSON file (either the recipe itself or a file with a ``recipe`` entry)."""
    with open(fn) as fp:
        d = json.load(fp)
    if isinstance(d, dict) and "recipe" in d:
        d = d["recipe"]
    return validate_recipe(d)


def parse_recipe(spec: str, base: Mapping[str, object] | None = None) -> Recipe:
    """Parse a ``--cut-recipe`` specification.

    ``spec`` is a JSON file, ``'key=value,key=value,...'`` on top of ``base``
    (default: the packaged recipe), or ``'FILE,key=value,...'`` (file plus overrides).
    """
    r: Recipe = dict(base) if base else default_recipe()
    parts = [p.strip() for p in str(spec).replace(";", ",").split(",") if p.strip()]
    if parts and "=" not in parts[0]:
        r = load_recipe_file(parts[0])
        parts = parts[1:]
    for p in parts:
        k, v = p.split("=", 1)
        k = k.strip()
        if k not in RECIPE_FIELDS:
            raise ValueError(f"--cut-recipe: unknown field {k!r} (fields: {RECIPE_FIELDS})")
        r[k] = v.strip()
    return validate_recipe(r)


def route_kwargs(r: Mapping[str, object]) -> dict[str, str]:
    """Keyword arguments of :meth:`mfa.surface.Hemi.seam_paths` for a recipe."""
    return {k: str(r[k]) for k in ROUTE_FIELDS}


def recipe_str(r: Mapping[str, object]) -> str:
    """``'key=value,...'`` form (usable as ``--cut-recipe``)."""
    return ",".join(f"{k}={r[k]}" for k in RECIPE_FIELDS)


def recipe_code(r: Mapping[str, object]) -> str:
    """Filesystem-safe short code of a recipe (used in scratch patch names)."""

    def code(field: str, val: object) -> str:
        val = str(val)
        c = RECIPE_CODES.get(field, {}).get(val)
        if c is None:
            c = hashlib.md5(val.encode()).hexdigest()[:5]
        return c

    s = (f"w{int(r['slit_width'])}_{code('temporal_src', r['temporal_src'])}_"  # type: ignore[call-overload]
         f"{code('temporal_dst', r['temporal_dst'])}_{code('temporal_corridor', r['temporal_corridor'])}_"
         f"{code('temporal_extend', r['temporal_extend'])}")
    k = code("calcarine_extend", r["calcarine_extend"])
    if k:
        s += "_" + k
    return s.replace("/", "-")


def recipe_key(r: Mapping[str, object]) -> tuple[str, ...]:
    """Hashable identity of a recipe."""
    return tuple(str(r[k]) for k in RECIPE_FIELDS)


def parse_slit_width(val: str | None) -> dict[str, int | str | None]:
    """Parse ``--slit-width``.

    ``None`` -> both hemispheres None (keep the recipe value); ``'2'`` -> both 2;
    ``'auto'`` -> both ``'auto'``; ``'lh=1,rh=auto'`` -> per hemisphere (a
    hemisphere not listed keeps the recipe value).
    """
    if val is None:
        return {"lh": None, "rh": None}

    def one(x: str) -> int | str:
        x = x.strip().lower()
        if x == "auto":
            return "auto"
        w = int(x)
        if w < 1:
            raise ValueError("slit width must be >= 1 (or 'auto')")
        return w

    txt = str(val).strip()
    out: dict[str, int | str | None] = {"lh": None, "rh": None}
    if "=" not in txt:
        w = one(txt)
        return {h: w for h in out}
    for part in txt.replace(";", ",").split(","):
        if not part.strip():
            continue
        h, x = part.split("=", 1)
        h = h.strip().lower()
        if h not in out:
            raise ValueError(f"unknown hemisphere {h!r} in {val!r} (use lh=..,rh=..)")
        out[h] = one(x)
    return out
