"""Cut recipes: one set of landmark rules shared by every hemisphere of every subject.

A recipe is a dictionary of twelve fields (:data:`RECIPE_FIELDS`).  Eleven of them
define *where* the cuts run, using only parcel membership and geodesic distance
on the surface, so they are mirror-symmetric and subject-independent by
construction.  No rule may name a coordinate axis: "the most anterior vertex"
depends on how the head lies in the scanner and is not the same anatomy on a
different brain, so every endpoint is a parcel centroid, a parcel-border
centroid, or a geodesic anchor inside a named parcel.

``frontal_src``
    rim entry of the frontal cut: ``nearest_<landmark>`` (the medial-wall rim
    vertex nearest a frontal landmark, :data:`FRONTAL_SRC_OPTIONS`).
``frontal_dst``
    tip of the frontal cut inside cortex: a geodesic anchor
    (:data:`FRONTAL_DST_OPTIONS`).
``frontal_corridor``
    parcels the frontal cut may run through: ``frontal`` (OFC, lat_PFC, ACgG) or
    ``medial`` (OFC, ACgG); ``band`` in border mode (below).
``frontal_route`` / ``temporal_route``
    how the seam runs between its ends.  ``path``: corridor-restricted,
    sulcus-preferring shortest path between the two landmarks.
    ``border:<A>_<B>`` (*anchored path*, the preferred border mode): shortest
    path between the same two landmarks inside A | B (plus the wall-adjacent
    vertices) with edge cost ``length * (1 + border_pull * d)``, ``d`` = geodesic
    distance (mm) of the edge from the A/B border, so the seam is pulled onto
    the border without tracing its jagged vertex chain; the corridor is implied
    (``*_corridor = pair``).  ``border_trace:<A>_<B>``: the literal border chain
    from its end nearest the medial-wall rim to the border vertex nearest the
    ``*_dst`` landmark (or the far end for ``*_dst = border_end``), gaps bridged
    inside A | B, smoothed to the shortest line inside a one-ring band
    (``*_src = border_rim``, ``*_corridor = band``); kept for comparison.
``border_pull``
    ``k`` of the anchored cost, per mm (:data:`BORDER_PULL_OPTIONS`; default
    0.5: two millimetres off the border doubles the cost).  Ignored, and
    normalised to the default, when no route is anchored.
``temporal_src``
    rim entry of the temporal seam: ``nearest_<landmark>`` = the medial-wall rim
    vertex nearest an anatomical landmark (``<A>_<B>_border`` or
    ``<P1>_<P2>.._centroid``).
``temporal_dst``
    destination of the temporal seam: a geodesic anchor ``<P>_far_<Q>`` = the
    vertex of parcel P geodesically farthest from the P/Q border.
``temporal_corridor``
    parcels the temporal seam may run through: ``medial`` (MTL, Amy, TG, ITC),
    ``lateral`` (TG, ITC only) or ``border`` (medial corridor with edges inside
    Amy/MTL penalised by :data:`BORDER_PENALTY`).
``temporal_extend``
    optional posterior continuation of the seam inside TG/ITC to a landmark
    (:data:`TEMPORAL_EXTEND_OPTIONS`); ``none`` = no extension.
``calcarine_extend``
    optional continuation of the calcarine cut within V1/V2-V4
    (:data:`CALCARINE_EXTEND_OPTIONS`); ``none`` = stop at the occipital pole.

The twelfth field, ``slit_width`` (vertex rings removed around the route), is a
technical mesh parameter and not part of the anatomical standard: the recipe
value is only a default, and the standard is the narrowest width that passes
every quality gate on each hemisphere (``--slit-width auto``).

The calcarine cut has no route field: it always runs from the rim vertex
nearest the V1 centroid to the occipital pole inside V1 | V2-V4
(``calcarine_extend`` optionally continues it).

Recipe files carry ``"format": "cut_recipe/2"``.  A ``cut_recipe/1`` file (no
frontal fields) still loads: its frontal fields are filled with
the default frontal rule.  Every landmark is coordinate-free: a parcel or border
centroid, or a geodesic anchor (``<P>_far_<Q>``, ``<A>_<B>_border_far_<Q|wall>``).
:func:`is_coordinate_free` decides this, :func:`validate_recipe` enforces it on
every recipe that is loaded, and ``mfa tune-joint`` refuses a searched option
that does not satisfy it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping
from importlib import resources

log = logging.getLogger("mfa")

TEMPORAL_CORRIDORS = ("medial", "lateral", "border")
BORDER_PENALTY = 4.0  # 'border' corridor: edge cost multiplier inside Amy/MTL
#: Frontal-cut corridors: parcels the frontal seam may run through (plus the vertices adjacent to the wall).
FRONTAL_CORRIDORS = {"frontal": ("OFC", "lat_PFC", "ACgG"), "medial": ("OFC", "ACgG")}
#: Route modes: ``path`` (landmark-to-landmark shortest path), ``border:<A>_<B>`` (path anchored to the
#: A/B border) or ``border_trace:<A>_<B>`` (the border chain itself).
ROUTE_PATH = "path"
BORDER_PREFIX = "border:"
TRACE_PREFIX = "border_trace:"
#: Rim-entry / corridor / destination values implied by the border routes.
BORDER_RIM = "border_rim"    # *_src (trace): the border end nearest the medial-wall rim
BAND = "band"                # *_corridor (trace): one-ring band around the border chain
PAIR = "pair"                # *_corridor (anchored): the two parcels of the border (+ wall-adjacent vertices)
BORDER_END = "border_end"    # *_dst (border routes): the far end of the border
BAND_RINGS = 1               # trace: width (vertex rings) of the smoothing band around the border chain
BRIDGE_PENALTY = 8.0         # trace: cost multiplier of edges that leave the border (gap bridging)
#: Anchored path: edge cost = length * (1 + border_pull * distance-from-border [mm]).
BORDER_PULL_OPTIONS = (0.25, 0.5, 1.0)
BORDER_PULL_DEFAULT = 0.5
FRONTAL_BORDERS = ("OFC_lat_PFC", "OFC_ACgG", "lat_PFC_ACgG")
TEMPORAL_BORDERS = ("Amy_TG", "MTL_TG", "Amy_ITC", "TG_ITC", "MTL_ITC")
FRONTAL_ROUTE_OPTIONS = ((ROUTE_PATH,) + tuple(BORDER_PREFIX + p for p in FRONTAL_BORDERS)
                         + tuple(TRACE_PREFIX + p for p in FRONTAL_BORDERS))
TEMPORAL_ROUTE_OPTIONS = ((ROUTE_PATH,) + tuple(BORDER_PREFIX + p for p in TEMPORAL_BORDERS)
                          + tuple(TRACE_PREFIX + p for p in TEMPORAL_BORDERS))
#: Parcels whose bank of the temporal seam is compared between hemispheres / subjects.
SEAM_PARCELS = ("Amy", "MTL", "MPal", "TG", "ITC")
#: Parcels the temporal slit may run through (identifies the temporal slit of an existing patch).
TEMPORAL_CORRIDOR_PARCELS = ("MTL", "Amy", "TG", "ITC")
#: Spellings usable in landmark rules for parcels whose name contains '-'.
PARCEL_ALIASES = {"V2V4": "V2-V4"}

DEFAULT_FRONTAL = {"frontal_src": "nearest_OFC_ACgG_border",
                   "frontal_dst": "OFC_lat_PFC_border_far_wall",
                   "frontal_corridor": "frontal"}

TEMPORAL_SRC_OPTIONS = (
    "nearest_MTL_Amy_centroid",   # rim vertex nearest the centroid of MTL+Amy
    "nearest_Amy_TG_border",      # ... nearest the centroid of the Amy/TG border
    "nearest_Amy_MPal_border",    # ... nearest the Amy/MPal border
    "nearest_MPal_MTL_border",    # ... nearest the MPal/MTL border
    "nearest_Amy_centroid",       # ... nearest the centroid of Amy
)
TEMPORAL_DST_OPTIONS = (
    "TG_tip",                     # temporal pole: TG vertex farthest from every other cortical parcel
    "TG_far_ITC",                 # TG vertex farthest from the TG/ITC border
    "TG_far_STG/STSd",            # TG vertex farthest from the TG/STG-STSd border
    "Amy_far_MTL",                # Amy vertex farthest from the Amy/MTL border
    "MTL_far_ITC",                # MTL vertex farthest from the MTL/ITC border
)
TEMPORAL_EXTEND_OPTIONS = (
    "none",                       # no posterior extension
    "ITC_far_TG",                 # along TG|ITC to the ITC vertex farthest from the TG/ITC border
    "ITC_MT_border",              # ... to the ITC/MT border
    "ITC_V2-V4_border",           # ... to the ITC / V2-V4 border
)
CALCARINE_EXTEND_OPTIONS = (
    "none",                       # stop at the occipital pole
    "V2-V4_far_V1",               # continue within V1|V2-V4 to the V2-V4 vertex farthest from V1
)
FRONTAL_SRC_OPTIONS = (
    "nearest_OFC_ACgG_border",         # rim vertex nearest the centroid of the OFC/ACgG border
    "nearest_OFC_centroid",            # ... nearest the centroid of OFC
    "nearest_ACgG_centroid",           # ... nearest the centroid of ACgG
    "nearest_OFC_lat_PFC_border",      # ... nearest the centroid of the OFC/lat_PFC border
)
FRONTAL_DST_OPTIONS = (
    "OFC_lat_PFC_tip",                 # frontal pole: OFC+lat_PFC vertex farthest from every other parcel
    "OFC_lat_PFC_border_far_wall",     # frontal pole where the orbital and lateral surfaces meet
    "lat_PFC_far_motor",               # lat_PFC vertex farthest from the lat_PFC/motor border
    "OFC_far_ACgG",                    # OFC vertex farthest from the OFC/ACgG border
    "ACgG_far_PMC",                    # ACgG vertex farthest from the ACgG/PMC border
)
FRONTAL_CORRIDOR_OPTIONS = tuple(FRONTAL_CORRIDORS)
RECIPE_FIELDS = ("slit_width", "temporal_src", "temporal_dst", "temporal_corridor",
                 "temporal_extend", "calcarine_extend", "frontal_src", "frontal_dst", "frontal_corridor",
                 "temporal_route", "frontal_route", "border_pull")
ROUTE_FIELDS = RECIPE_FIELDS[1:]
#: Route fields that are numbers, not option strings (kept numeric by :func:`route_kwargs`).
NUMERIC_ROUTE_FIELDS = ("border_pull",)
FRONTAL_FIELDS = ("frontal_src", "frontal_dst", "frontal_corridor", "frontal_route")
TEMPORAL_FIELDS = ("temporal_src", "temporal_dst", "temporal_corridor", "temporal_route")
#: Value of the route-mode fields when a recipe (any format) does not list them.
ROUTE_DEFAULTS = {"temporal_route": ROUTE_PATH, "frontal_route": ROUTE_PATH, "border_pull": BORDER_PULL_DEFAULT}
RECIPE_FORMAT = "cut_recipe/2"
#: Recipe file formats this module reads (older ones are upgraded on load).
RECIPE_FORMATS = ("cut_recipe/1", RECIPE_FORMAT)
#: Landmark options of every route field (searched by ``mfa tune-joint`` and its refinement).
FIELD_OPTIONS = {
    "temporal_src": TEMPORAL_SRC_OPTIONS, "temporal_dst": TEMPORAL_DST_OPTIONS,
    "temporal_corridor": TEMPORAL_CORRIDORS, "temporal_extend": TEMPORAL_EXTEND_OPTIONS,
    "calcarine_extend": CALCARINE_EXTEND_OPTIONS, "frontal_src": FRONTAL_SRC_OPTIONS,
    "frontal_dst": FRONTAL_DST_OPTIONS, "frontal_corridor": FRONTAL_CORRIDOR_OPTIONS,
    "temporal_route": TEMPORAL_ROUTE_OPTIONS, "frontal_route": FRONTAL_ROUTE_OPTIONS,
    "border_pull": BORDER_PULL_OPTIONS,
}
#: Slit widths tried, in order, by the ``auto`` width policy.
AUTO_SLIT_WIDTHS = (1, 2, 3)

# short codes used in scratch patch names (<base>_w2_Aa_TGa_lat_e0)
def _auto_codes(options, prefix=""):
    """Short, stable codes for patch names: initials of the option string."""
    out = {}
    for o in options:
        t = str(o).replace("nearest_", "").replace("_border", "b").replace("_far_", "f")
        t = "".join(p[:2] for p in re.split(r"[_/\-]", t) if p)
        out[str(o)] = prefix + t[:8]
    return out


RECIPE_CODES = {
    "temporal_src": _auto_codes(TEMPORAL_SRC_OPTIONS),
    "temporal_dst": _auto_codes(TEMPORAL_DST_OPTIONS),
    "temporal_corridor": {"medial": "med", "lateral": "lat", "border": "bor", BAND: "bnd", PAIR: "pr"},
    "temporal_extend": _auto_codes(TEMPORAL_EXTEND_OPTIONS, "e"),
    "calcarine_extend": _auto_codes(CALCARINE_EXTEND_OPTIONS, "k"),
    "frontal_src": _auto_codes(FRONTAL_SRC_OPTIONS, "f"),
    "frontal_dst": _auto_codes(FRONTAL_DST_OPTIONS, "d"),
    "frontal_corridor": {"frontal": "", "medial": "m", BAND: "", PAIR: ""},
    "border_pull": {"0.5": "", "0.25": "k025", "1.0": "k1"},
}
RECIPE_CODES["temporal_dst"][BORDER_END] = "E"
RECIPE_CODES["frontal_dst"][BORDER_END] = "E"
RECIPE_CODES["temporal_src"][BORDER_RIM] = "R"
RECIPE_CODES["temporal_dst"][BORDER_END] = "E"
RECIPE_CODES["temporal_corridor"][BAND] = ""
RECIPE_CODES["temporal_corridor"][PAIR] = ""

Recipe = dict[str, object]


def packaged_recipe_path() -> str:
    """Path of the recipe shipped with the package (``mfa/data/cut_recipe.json``)."""
    return str(resources.files("mfa").joinpath("data", "cut_recipe.json"))


def default_recipe() -> Recipe:
    """The packaged default recipe as a validated dictionary."""
    return load_recipe_file(packaged_recipe_path())


#: Landmark forms allowed as a *point* (rim-entry rules): centroids and geodesic anchors.
POINT_SUFFIXES = ("_centroid", "_border", "_tip")
#: Landmark forms allowed as a *vertex* (seam destinations): geodesic anchors only.
VERTEX_SUFFIXES = ("_tip",)
#: A landmark is coordinate-free if it is a centroid, a border centroid, or a geodesic anchor
#: (``<P>_far_<Q>`` / ``<A>_<B>_border_far_<Q|wall>``).  No rule may use a coordinate axis.
GEODESIC_RE = re.compile(r"^.+_far_.+$")
#: Destination of the calcarine cut: the occipital pole (V1 vertex farthest from V1/V2-V4).
CALCARINE_DST = "V1_far_V2-V4"
#: Destination of the optional cingulate cut: the outer end of the ACgG/PMC border.
CINGULATE_DST = "ACgG_PMC_border_far_wall"



#: Option values that name no landmark at all (corridors, routes, numbers, 'none').
#: Values that are not landmarks at all: route modes and the corridor / endpoint names a route
#: implies.  BORDER_RIM (the end of the border nearest the medial wall) and BORDER_END (its far
#: end) are positions on a named border, so they are coordinate-free like any other landmark.
_NON_LANDMARK = {"none", ROUTE_PATH, BORDER_RIM, BORDER_END, BAND, PAIR,
                 *TEMPORAL_CORRIDORS, *FRONTAL_CORRIDORS}


def is_coordinate_free(field: str, value: object) -> bool:
    """True if a recipe option is defined without any coordinate axis.

    Cut rules may only use the parcellation and distances on the surface: parcel or
    parcel-border centroids (``<P>_centroid``, ``<A>_<B>_border``), geodesic anchors
    (``<P>_far_<Q>``, ``<A>_<B>_border_far_<Q|wall>``), corridors, route modes and numbers.
    Anything selecting a vertex by an anterior/posterior or dorsal/ventral extreme
    (the old ``<P>_ant`` / ``<P>_post`` forms) is rejected.
    """
    v = str(value)
    if field in ("slit_width", "border_pull") or v in _NON_LANDMARK:
        return True
    if v.startswith(BORDER_PREFIX) or v.startswith(TRACE_PREFIX):
        return True
    core = v[len("nearest_"):] if v.startswith("nearest_") else v
    return (bool(GEODESIC_RE.match(core)) or core.endswith("_centroid")
            or core.endswith("_border") or core.endswith("_tip"))


def _valid_landmark(spec: str, suffixes: tuple[str, ...]) -> bool:
    """True for a coordinate-free landmark: a geodesic anchor or one of `suffixes`."""
    return bool(GEODESIC_RE.match(spec)) or any(spec.endswith(x) for x in suffixes)





def is_anchored_route(spec: object) -> bool:
    """True for a ``border:<A>_<B>`` (anchored path) route mode."""
    return str(spec).startswith(BORDER_PREFIX)


def is_trace_route(spec: object) -> bool:
    """True for a ``border_trace:<A>_<B>`` route mode."""
    return str(spec).startswith(TRACE_PREFIX)


def is_border_route(spec: object) -> bool:
    """True for either border route mode (anchored or trace)."""
    return is_anchored_route(spec) or is_trace_route(spec)


def border_pair(spec: object) -> str:
    """``'A_B'`` of a ``border:A_B`` / ``border_trace:A_B`` route (the surface splits the pair, it knows the names)."""
    if not is_border_route(spec):
        raise ValueError(f"not a border route: {spec!r}")
    pair = str(spec)[len(TRACE_PREFIX if is_trace_route(spec) else BORDER_PREFIX):]
    if "_" not in pair.strip("_") or not pair:
        raise ValueError(f"route {spec!r}: expected 'border:<A>_<B>' or 'border_trace:<A>_<B>'")
    return pair


def canonicalise_routes(r: Mapping[str, object]) -> Recipe:
    """Set the fields a route mode implies: trace -> ``*_src = border_rim``, ``*_corridor = band``;
    anchored -> ``*_corridor = pair``; no anchored route -> ``border_pull`` = default.

    Used by the joint search so that the option product does not multiply a
    border route by options that it does not use.
    """
    out: Recipe = dict(r)
    anchored = False
    for cut in ("temporal", "frontal"):
        rt = out.get(f"{cut}_route", ROUTE_PATH)
        if is_trace_route(rt):
            out[f"{cut}_src"], out[f"{cut}_corridor"] = BORDER_RIM, BAND
        elif is_anchored_route(rt):
            out[f"{cut}_corridor"] = PAIR
            anchored = True
    if not anchored:
        out["border_pull"] = BORDER_PULL_DEFAULT
    return out


def upgrade_recipe(r: Mapping[str, object], fmt: str | None = None) -> Recipe:
    """Fill the fields a ``cut_recipe/1`` recipe lacks with their documented legacy values.

    A recipe without the frontal fields (or a file declaring ``cut_recipe/1``) gets
    the default frontal rule; nothing else is changed.  Raises ``ValueError`` on an
    unknown ``fmt`` or on a ``cut_recipe/2`` recipe with missing frontal fields.
    """
    if fmt is not None and fmt not in RECIPE_FORMATS:
        raise ValueError(f"recipe: unknown format {fmt!r} (known: {RECIPE_FORMATS})")
    out: Recipe = dict(r)
    missing = [k for k in FRONTAL_FIELDS if k not in out and k in DEFAULT_FRONTAL]
    if missing:
        for k in missing:
            out[k] = DEFAULT_FRONTAL[k]
        log.info("recipe: %s has no frontal fields; using the default frontal rule (%s -> %s)",
                 fmt or "cut_recipe/1", DEFAULT_FRONTAL["frontal_src"], DEFAULT_FRONTAL["frontal_dst"])
    return out


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
    for cut in ("temporal", "frontal"):
        rt = str(out.get(f"{cut}_route", ROUTE_PATH))
        if rt != ROUTE_PATH:
            border_pair(rt)  # raises on a malformed spec
        out[f"{cut}_route"] = rt
    t_border = is_border_route(out["temporal_route"])
    s = str(out["temporal_src"])
    if is_trace_route(out["temporal_route"]):
        if s != BORDER_RIM or str(out["temporal_corridor"]) != BAND:
            raise ValueError(f"recipe: temporal_route {out['temporal_route']!r} requires temporal_src={BORDER_RIM} "
                             f"and temporal_corridor={BAND} (the border end nearest the rim, the band around the "
                             "border chain)")
    else:
        if is_anchored_route(out["temporal_route"]):
            if str(out["temporal_corridor"]) != PAIR:
                raise ValueError(f"recipe: temporal_route {out['temporal_route']!r} requires temporal_corridor={PAIR} "
                                 "(the two parcels of the border)")
        elif out["temporal_corridor"] not in TEMPORAL_CORRIDORS:
            raise ValueError(f"recipe: temporal_corridor {out['temporal_corridor']!r} not in {TEMPORAL_CORRIDORS}")
        if not (s.startswith("nearest_") and _valid_landmark(s, POINT_SUFFIXES)):
            raise ValueError(f"recipe: temporal_src {s!r}: 'nearest_<landmark>' with landmark "
                             "'<A>_<B>_border' | '<P1>_<P2>.._centroid' | '<P>_far_<Q>'")
    d = str(out["temporal_dst"])
    if not ((t_border and d == BORDER_END) or _valid_landmark(d, VERTEX_SUFFIXES)):
        raise ValueError(f"recipe: temporal_dst {d!r}: a geodesic anchor '<P>_far_<Q>'"
                         + (f" or {BORDER_END}" if t_border else ""))
    out["temporal_src"], out["temporal_corridor"] = str(out["temporal_src"]), str(out["temporal_corridor"])
    for field, options in (("temporal_extend", TEMPORAL_EXTEND_OPTIONS),
                           ("calcarine_extend", CALCARINE_EXTEND_OPTIONS)):
        e = str(out[field])
        if e.startswith("mm:") or e.startswith("ITC_frac_"):
            raise ValueError(f"recipe: {field} {e!r}: millimetre and fraction-along-an-axis extensions are "
                             "not anatomical landmarks and are not supported; use one of "
                             f"{options} or a landmark ('<A>_<B>_border' | '<P>_far_<Q>')")
        if not (e in options or _valid_landmark(e, POINT_SUFFIXES)):
            raise ValueError(f"recipe: {field} {e!r} not in {options}")
        out[field] = e
    fs, fd, fc = str(out["frontal_src"]), str(out["frontal_dst"]), str(out["frontal_corridor"])
    if is_trace_route(out["frontal_route"]):
        if fs != BORDER_RIM or fc != BAND:
            raise ValueError(f"recipe: frontal_route {out['frontal_route']!r} requires frontal_src={BORDER_RIM} and "
                             f"frontal_corridor={BAND}")
        if not (fd == BORDER_END or _valid_landmark(fd, VERTEX_SUFFIXES)):
            raise ValueError(f"recipe: frontal_dst {fd!r}: a geodesic anchor "
                             f"('<P>_far_<Q>', '<A>_<B>_border_far_<Q|wall>') or {BORDER_END}")
    elif is_anchored_route(out["frontal_route"]):
        if fc != PAIR:
            raise ValueError(f"recipe: frontal_route {out['frontal_route']!r} requires frontal_corridor={PAIR}")
        if not (fs.startswith("nearest_") and _valid_landmark(fs, POINT_SUFFIXES)):
            raise ValueError(f"recipe: frontal_src {fs!r}: an anchored route needs 'nearest_<landmark>'")
        if not (fd == BORDER_END or _valid_landmark(fd, VERTEX_SUFFIXES)):
            raise ValueError(f"recipe: frontal_dst {fd!r}: a geodesic anchor "
                             f"('<P>_far_<Q>', '<A>_<B>_border_far_<Q|wall>') or {BORDER_END}")
    else:
        if not (fs.startswith("nearest_") and _valid_landmark(fs, POINT_SUFFIXES)):
            raise ValueError(f"recipe: frontal_src {fs!r}: 'nearest_<landmark>' with landmark "
                             "'<A>_<B>_border' | '<P1>_<P2>.._centroid' | '<P>_far_<Q>'")
        if not _valid_landmark(fd, VERTEX_SUFFIXES):
            raise ValueError(f"recipe: frontal_dst {fd!r}: a geodesic anchor "
                             "('<P>_far_<Q>' | '<A>_<B>_border_far_<Q|wall>')")
        if fc not in FRONTAL_CORRIDORS:
            raise ValueError(f"recipe: frontal_corridor {fc!r} not in {tuple(FRONTAL_CORRIDORS)}")
    out["frontal_src"], out["frontal_dst"], out["frontal_corridor"] = fs, fd, str(out["frontal_corridor"])
    try:
        out["border_pull"] = float(out["border_pull"])  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"recipe: border_pull {out['border_pull']!r} must be a number (per mm)") from None
    if out["border_pull"] <= 0:  # type: ignore[operator]
        raise ValueError("recipe: border_pull must be > 0")
    # the standard in one line: no cut rule may name a coordinate axis
    for k in ROUTE_FIELDS:
        if not is_coordinate_free(k, out[k]):
            raise ValueError(f"recipe: {k} {out[k]!r} is not coordinate-free.  Cut rules may only use the "
                             "parcellation and distances on the surface: '<P1>_<P2>.._centroid', "
                             "'<A>_<B>_border', '<P>_far_<Q>', '<A>_<B>_border_far_<Q|wall>', a corridor or a "
                             "route mode.  A rule that picks a vertex by an anterior/posterior or "
                             "dorsal/ventral extreme depends on how the head lies in the scanner and is not "
                             "the same anatomy on another brain.")
    shift = float(out.get("temporal_src_shift_mm", 0.0) or 0.0)  # type: ignore[arg-type]
    if shift != 0.0:
        raise ValueError("recipe: temporal_src_shift_mm is not a landmark rule and is not supported")
    return {k: out[k] for k in RECIPE_FIELDS}


def load_recipe_file(fn: str) -> Recipe:
    """Load a recipe JSON file (either the recipe itself or a file with a ``recipe`` entry).

    A ``cut_recipe/1`` file is upgraded with :func:`upgrade_recipe` (legacy frontal rule).
    """
    with open(fn) as fp:
        d = json.load(fp)
    fmt = None
    if isinstance(d, dict) and "recipe" in d:
        fmt = d.get("format")
        d = d["recipe"]
    r = upgrade_recipe(d, fmt)
    return validate_recipe(r)


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


def route_kwargs(r: Mapping[str, object]) -> dict[str, object]:
    """Keyword arguments of :meth:`mfa.surface.Hemi.seam_paths` for a recipe.

    Numeric fields stay numeric: ``border_pull`` is the coefficient of the anchored-route edge
    cost and must not be passed as a string.
    """
    out: dict[str, object] = {k: str(r[k]) for k in ROUTE_FIELDS if k not in NUMERIC_ROUTE_FIELDS}
    for k in NUMERIC_ROUTE_FIELDS:
        if k in ROUTE_FIELDS:
            out[k] = float(r[k])  # type: ignore[arg-type]
    return out


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

    t_route = str(r.get("temporal_route", ROUTE_PATH))
    t_src = code("temporal_src", r["temporal_src"]) + code("temporal_route", t_route)
    t_corr = "" if is_border_route(t_route) else code("temporal_corridor", r["temporal_corridor"])
    parts = [f"w{int(r['slit_width'])}", t_src, code("temporal_dst", r["temporal_dst"]),  # type: ignore[call-overload]
             t_corr, code("temporal_extend", r["temporal_extend"]), code("calcarine_extend", r["calcarine_extend"])]
    s = "_".join(p for p in parts if p)
    fr = {f: str(r.get(f, dict(DEFAULT_FRONTAL, **ROUTE_DEFAULTS)[f])) for f in FRONTAL_FIELDS}
    if fr != {f: str(dict(DEFAULT_FRONTAL, **ROUTE_DEFAULTS)[f]) for f in FRONTAL_FIELDS}:
        f_src = code("frontal_src", fr["frontal_src"]) + code("frontal_route", fr["frontal_route"])
        f_corr = "" if is_border_route(fr["frontal_route"]) else code("frontal_corridor", fr["frontal_corridor"])
        s += f"_f{f_src}-{code('frontal_dst', fr['frontal_dst'])}{f_corr}"
    if any(is_anchored_route(r.get(f, ROUTE_PATH)) for f in ("temporal_route", "frontal_route")):
        k = code("border_pull", f"{float(r.get('border_pull', BORDER_PULL_DEFAULT)):g}")  # type: ignore[arg-type]
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
